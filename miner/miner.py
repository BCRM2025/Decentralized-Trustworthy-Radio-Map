import logging
from array import array
import numpy as np
from numpy import linalg as LA
from sensor import Sensor
from Crypto.Signature import PKCS1_v1_5
from Crypto.Hash import SHA256
from collections import deque
import math

import global_var
from consensus import Consensus
from data import Block, Message
from external import I
from functions import for_name

from ._consts import FLOODING, SPEC_TARGETS, OUTER_RCV_MSG, SELF_GEN_MSG, SYNC_LOC_CHAIN
from .network_interface import NetworkInterface, NICWithoutTp, NICWithTp

logger = logging.getLogger(__name__)


class Miner(object):
    def __init__(self, miner_id, consensus_params:dict, max_block_capacity:int = 0,
                 disable_dataitem_queue=False, K=5, T=3, b=2.5):
        self.miner_id = miner_id #矿工ID
        self._isAdversary = False
        #共识相关
        self.consensus:Consensus = for_name(
            global_var.get_consensus_type())(miner_id, consensus_params)# 共识

        #输入内容相关
        self.input_tape = []
        self.round = -1
        self.query_positions = None
        self.sensor_data = {}  # Temporary storage for current round
        self.valid_database = {}  # Persistent storage for valid data
        self.local_data = {}  # {sensor_ID: trust_score}
        self.sent_data = np.array([]).reshape(-1, 5)
        self.map = np.array([]).reshape(-1, 3)
        self.flag = 1
        self.gap = 10000
        self.cnt = 0
        self.time_now = 0

        # Hyperparameters
        self.K = K
        self.T = T
        self.b = b
        # 新增：邻居索引缓存
        self.neighbor_cache = {}  # {sensor_id: [neighbor_ids]}
        # 新增：邻居历史数据
        self.neighbor_history = {i: deque(maxlen=self.T) for i in
                                 range(17271)}  # {sensor_id: deque([[x, y, value, label, timestamp], ...])}
        # 新增：用于时间一致性的历史值
        self.sensor_history = {}  # {sensor_ID: deque(maxlen=5)}
        for i in range(17271):  # 为所有传感器初始化历史记录
            self.sensor_history[i] = deque(maxlen=10)
        # 新增：区块链的异常值日志
        self.outlier_log = []
        # 新增：异常检测开关
        self.enable_detection = False  # 默认启用异常检测

        # Define output file path for evaluation results
        RESULT_PATH = global_var.get_result_path()
        RESULT_PATH.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        self.output_file = RESULT_PATH / f'detection_metrics_miner_{miner_id}.txt'
        # Initialize output file
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(f"Detection Results (Miner {miner_id})\n")
            f.write("==============================\n")

        #网络接口
        self._NIC:NetworkInterface =  None
        self.__forwarding_targets = None
        self.receive_history = dict()
        # maximum data items in a block
        self.max_block_capacity = max_block_capacity
        if self.max_block_capacity > 0 and not disable_dataitem_queue:
            self.dataitem_queue = array('Q')
        #保存矿工信息
        CHAIN_DATA_PATH=global_var.get_chain_data_path()
        with open(CHAIN_DATA_PATH / f'chain_data{str(self.miner_id)}.txt','a') as f:
            print(f"Miner {self.miner_id}\n"
                  f"consensus_params: {consensus_params}", file=f)
    
    def get_local_chain(self):
        return self.consensus.local_chain
    
    def in_local_chain(self, block:Block):
        return self.consensus.in_local_chain(block)

    def has_received(self, block:Message):
        return self.consensus.has_received(block)

    def _join_network(self, network):
        """初始化网络接口"""
        if network.withTopology:
            self._NIC = NICWithTp(self)
        else:
            self._NIC = NICWithoutTp(self)
        self._NIC.nic_join_network(network)
    
    @property
    def network_has_topology(self):
        """网络是否有拓扑结构"""
        return isinstance(self._NIC, NICWithTp)

    @property
    def neighbors(self):
        return self._NIC._neighbors
        
    def set_adversary(self, _isAdversary:bool):
        '''
        设置是否为对手节点
        _isAdversary=True为对手节点
        '''
        self._isAdversary = _isAdversary
    
    def set_forwarding_targets(self, forwarding_targets:list):
        '''
        设置自动转发消息的目标节点列表，必须是邻居列表的子集
        '''
        self.__forwarding_targets = forwarding_targets

    def receive(self, source:int, msg: Message):
        '''处理接收到的消息，直接调用consensus.receive'''
        rcvSuccess = self.consensus.receive_filter(msg)
        if not rcvSuccess:
            return rcvSuccess
        else:
            self.receive_history[msg.name] = source
        if self.__forwarding_targets is None:
            self.forward([msg], OUTER_RCV_MSG)
        elif len(self.__forwarding_targets) > 0:
            self.forward([msg], OUTER_RCV_MSG, forward_strategy=SPEC_TARGETS,
                         spec_targets=self.__forwarding_targets)

        return rcvSuccess
    
       
    def forward(self, msgs:list[Message], msg_source_type, forward_strategy:str=FLOODING, 
                spec_targets:list=None, syncLocalChain = False):
        """将消息转发给其他节点

        args:
            msgs: list[Message] 需要转发的消息列表
            msg_source_type: str 消息来源类型, SELF_GEN_MSG表示由本矿工产生, OUTER_RCV_MSG表示由网络接收
            forward_strategy: str 消息转发策略
            spec_targets: list[int] 如果forward_strategy为SPECIFIC, 则spec_targets为转发的目标节点列表
            syncLocalChain: bool 是否向邻居同步本地链，尽量在产生新区块时同步
        
        """
        if msg_source_type != SELF_GEN_MSG and msg_source_type != OUTER_RCV_MSG:
            raise ValueError("Message type must be SELF or OUTER")
        logger.info("M%d: forwarding %s, type %s, strategy %s", self.miner_id, 
                    str([msg.name for msg in msgs] if len(msgs)>0 else [""]), msg_source_type, forward_strategy)
        for msg in msgs:
            self._NIC.append_forward_buffer(msg, msg_source_type, forward_strategy, spec_targets, syncLocalChain)

    
    def launch_consensus(self, input, sent_data, Map, state, round):
        '''开始共识过程

        return:
            new_msg 由共识类产生的新消息，没有就返回None type:list[Message]/None
            msg_available 如果有新的消息产生则为True type:Bool
        '''
        new_msgs, msg_available = self.consensus.consensus_process(
            self._isAdversary, input, sent_data, Map, state, round)
        if new_msgs is not None:
            # new_msgs.append(Message("testMsg", 1))
            self.forward(new_msgs, SELF_GEN_MSG, syncLocalChain = True)
        return new_msgs, msg_available  # 返回挖出的区块，
        

    def BackboneProtocol(self, round, state):
        _, chain_update = self.consensus.local_state_update()
        input = I(round, self.input_tape)  # I function
        if self.max_block_capacity > 0 and getattr(self, 'dataitem_queue', None) is not None:
            # exclude dataitems in updated blocks
            if len(chain_update) > 0:
                dataitem_exclude = set()
                for block in chain_update:
                    block:Consensus.Block
                    dataitem_exclude.update(array('Q', block.blockhead.content))
                self.dataitem_queue = array('Q', [x for x in self.dataitem_queue if x not in dataitem_exclude])
            self.dataitem_queue.frombytes(input)
            if len(self.dataitem_queue) > 10 * self.max_block_capacity:
                # drop the oldest data items if the queue is longer than 2 * max_block_capacity
                self.dataitem_queue.pop(0)
            input = self.dataitem_queue[:self.max_block_capacity].tobytes()

        new_msgs, msg_available = self.launch_consensus(input, self.sent_data, self.map, state, round)

        if msg_available:
            # remove the data items in the new block from dataitem_queue
            if self.max_block_capacity > 0 and getattr(self, 'dataitem_queue', None) is not None:
                self.dataitem_queue = self.dataitem_queue[self.max_block_capacity:]
            return new_msgs
        return None  #  如果没有更新 返回空告诉environment回合结束
        

    def set_detection_enabled(self, enabled: bool):
        """Enable or disable anomaly detection."""
        self.enable_detection = enabled


    def evaluate_detection(self, true_labels, detected_labels, round_num):
        """Evaluate detection performance and save to file."""
        true_labels = np.array(true_labels, dtype=int)
        detected_labels = np.array(detected_labels, dtype=int)

        # Calculate confusion matrix
        TP = np.sum((true_labels == 1) & (detected_labels == 1))  # True Positives
        TN = np.sum((true_labels == 0) & (detected_labels == 0))  # True Negatives
        FP = np.sum((true_labels == 0) & (detected_labels == 1))  # False Positives
        FN = np.sum((true_labels == 1) & (detected_labels == 0))  # False Negatives

        # Calculate metrics
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0  # True Positive Rate (Recall)
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision
        F1 = 2 * (Precision * TPR) / (Precision + TPR) if (Precision + TPR) > 0 else 0  # F1 Score

        # Save to file
        with open(self.output_file, 'a+', encoding='utf-8') as f:
            f.write(f"Round {round_num}\n")
            f.write(f"Detection Enabled: {self.enable_detection}\n")
            f.write(f"TPR (Recall): {TPR:.4f}\n")
            f.write(f"FPR (False Positive Rate): {FPR:.4f}\n")
            f.write(f"Precision: {Precision:.4f}\n")
            f.write(f"F1 Score: {F1:.4f}\n")
            f.write(f"True Anomalies: {np.sum(true_labels)}\n")
            f.write(f"Detected Anomalies: {np.sum(detected_labels)}\n")
            f.write(f"True Labels Number: {true_labels.shape}\n")
            f.write(f"Detected Labels Number: {detected_labels.shape}\n")
            f.write(f"Valid Database Size: {len(self.valid_database)}\n")
            # f.write("Valid Database:\n")
            # if self.valid_database:
            #     for sensor_id in sorted(self.valid_database.keys()):
            #         data = self.valid_database[sensor_id]
            #         f.write(
            #             f"Sensor ID: {sensor_id}, Data: [x={data[0]:.1f}, y={data[1]:.1f}, value={data[2]:.1f}, label={data[3]}, timestamp={data[4]:.1f}]\n")
            # else:
            #     f.write("Valid Database: Empty\n")
            f.write("--------------------\n")


    def handle_anomaly(self, sensor_id, timestamp, reason):
        """Handle anomaly detection: update trust score and log."""
        self.local_data[sensor_id] = max(self.local_data[sensor_id] - 0.1, 0.0)
        self.outlier_log.append((sensor_id, timestamp, reason))


    def query_sensor(self, sensors: list[Sensor], query_positions, time_now):
        self.time_now = time_now
        self.cnt += 1

        # Initialize local_data and query_positions
        if self.flag == 1:
            self.query_positions = query_positions
            for i in range(len(sensors)):
                self.local_data[i] = 1.0  # trust_score=1
            self.flag = 0

        # Recover trust scores every 50 rounds
        if self.cnt % 50 == 0:
            for key in self.local_data:
                self.local_data[key] = min(self.local_data[key] + 0.1, 1.0)
            self.neighbor_cache.clear()  # Refresh neighbor cache

        # Stage 1: Collect data with signature verification
        temp_data = {}  # {sensor_id: [x, y, value, label, timestamp]}
        true_labels = []  # Store true labels (0=normal, 1=anomaly)
        detected_labels = []  # Store detection results (0=normal, 1=anomaly)
        for sensor in sensors:
            data_temp = sensor.upload_data()
            if not data_temp:
                continue

            sensor_data_temp = data_temp['sensing_data']
            timestamp = data_temp['timestamp']
            sig_temp = data_temp['sig']
            key_temp = data_temp['public_key']
            data = np.append(sensor_data_temp, timestamp)

            # Signature verification
            if global_var.get_sig():
                verifier = PKCS1_v1_5.new(key_temp)
                verified = verifier.verify(SHA256.new(sensor_data_temp.tobytes()), sig_temp)
                if not verified:
                    self.handle_anomaly(sensor.Sensor_ID, timestamp, "Signature verification failed")
                    continue

            temp_data[sensor.Sensor_ID] = data
            true_labels.append(sensor_data_temp[3])  # Record true label

        # Stage 2: Outlier detection
        if self.enable_detection:
            db_data = np.array(list(self.valid_database.values())) if self.valid_database else np.array([]).reshape(-1,
                                                                                                                    5)
            for sensor_id, data in temp_data.items():
                value = data[2]
                loc = data[:2]
                is_anomaly = False
                reason = ""

                # Get K=5 nearest neighbors
                if sensor_id in self.neighbor_cache:
                    neighbor_ids = self.neighbor_cache[sensor_id]
                else:
                    if db_data.size > 0:
                        neighbor_indices, _ = self.get_K_nearest_neighbors(db_data[:, :2], loc, self.K)
                        neighbor_ids = [list(self.valid_database.keys())[i] for i in neighbor_indices]
                        self.neighbor_cache[sensor_id] = neighbor_ids
                    else:
                        neighbor_ids = []

                # Perform four independent checks: current round + 3 historical rounds
                anomaly_flags = []
                reasons = []

                # Check 1: Current round (spatial consistency)
                if neighbor_ids and db_data.size > 0:
                    neighbor_values = [self.valid_database[nid][2] for nid in neighbor_ids if
                                       nid in self.valid_database]
                    if len(neighbor_values) >= 2:  # Relaxed for early rounds
                        median = np.median(neighbor_values)
                        std = np.std(neighbor_values) if len(neighbor_values) > 1 else 1.0
                        if abs(value - median) > self.b * std:
                            anomaly_flags.append(True)
                            reasons.append("Spatial inconsistency (current round)")
                        else:
                            anomaly_flags.append(False)
                    else:
                        anomaly_flags.append(False)

                # Checks 2-4: Historical rounds (temporal consistency)
                for round_idx in range(self.T):  # Check up to 3 historical rounds
                    neighbor_values = []
                    for nid in neighbor_ids:
                        if len(self.neighbor_history[nid]) > round_idx:
                            # Get value from round_idx-th historical round
                            hist_data = list(self.neighbor_history[nid])[
                                -(round_idx + 1)]  # Reverse to get older rounds
                            neighbor_values.append(hist_data[2])
                    if len(neighbor_values) >= 2:  # Relaxed for early rounds
                        median = np.median(neighbor_values)
                        std = np.std(neighbor_values) if len(neighbor_values) > 1 else 1.0
                        if abs(value - median) > self.b * std:
                            anomaly_flags.append(True)
                            reasons.append(f"Temporal inconsistency (historical round {round_idx + 1})")
                        else:
                            anomaly_flags.append(False)
                    else:
                        anomaly_flags.append(False)

                # Final anomaly decision
                is_anomaly = any(anomaly_flags)
                # if is_anomaly:
                #     reason = reasons[anomaly_flags.index(True)]  # Use first anomaly reason

                # Handle anomaly or store valid data
                detected_labels.append(1 if is_anomaly else 0)
                if is_anomaly:
                    self.handle_anomaly(sensor_id, data[4], reason)
                else:
                    self.valid_database[sensor_id] = data
                    self.neighbor_history[sensor_id].append(data.tolist())  # Store after validation
                    self.sensor_history[sensor_id].append(value)

        else:
            # Skip anomaly detection, store all data
            for sensor_id, data in temp_data.items():
                value = data[2]
                self.valid_database[sensor_id] = data
                self.neighbor_history[sensor_id].append(data.tolist())
                self.sensor_history[sensor_id].append(value)
                detected_labels.append(0)

        # Stage 3: Build sent_data from valid_database
        self.sent_data = np.array(list(self.valid_database.values())) if self.valid_database else np.array([]).reshape(
            -1, 5)

        # Stage 4: Evaluate detection performance
        if true_labels and detected_labels:
            self.evaluate_detection(true_labels, detected_labels, time_now)


    def pred_KNN(self, K, r):
        rx_locs_plot = self.query_positions
        N = rx_locs_plot.shape[0]
        pred_path_know = np.zeros((N, 1))
        for n in range(N):
            loc_cur = rx_locs_plot[n, :]
            neighbor_indices, neighbor_distances = self.get_K_nearest_neighbors(self.sent_data[:, :2], loc_cur, K)
            neighbor_distances = np.maximum(neighbor_distances, 10 ** -6)
            # distances_inverse = 1. / neighbor_distances
            distances_inverse = np.power(neighbor_distances, -5)
            a = self.time_now - self.sent_data[neighbor_indices][:, 4]
            # loc_weights = distances_inverse / np.sum(distances_inverse)
            # time_weights = math.e ** -a / np.sum(math.e ** -a)
            # weights = loc_weights * time_weights
            # weights = distances_inverse / np.sum(distances_inverse)
            # neighbor_labels = self.sent_data[neighbor_indices, 2]
            loc_weights = distances_inverse
            time_weights = math.e ** (-r * a)
            # weights = loc_weights * time_weights / np.sum(loc_weights * time_weights)
            weights = loc_weights * time_weights / (np.sum(loc_weights * time_weights) + 1e-300)
            neighbor_labels = self.sent_data[neighbor_indices, 2]
            pred_path_know[n, :] = np.matmul(weights, neighbor_labels)

        return pred_path_know


    def get_K_nearest_neighbors(self, locs_database, loc_cur, K):
        distances = LA.norm(locs_database - loc_cur, axis=1)
        neighbor_indices = np.argpartition(distances, K)[:K]
        neighbor_distances = distances[neighbor_indices]

        return neighbor_indices, neighbor_distances


    def data_process(self, r):
        Map = self.pred_KNN(4, r)
        self.map = np.concatenate((self.query_positions, Map), axis=1)
        # print('Number of pred_out:' + str(self.pred_output_KNN.shape[0]))
        return [list(row) for row in self.map]


    def get_map(self, Map):
        self.map = np.array(Map)


    def clear_tapes(self):
        # clear the input tape
        # self.input_tape = []
        # clear the communication tape
        self.consensus.receive_tape = []
        self._NIC._receive_buffer.clear()
        # self._NIC.clear_forward_buffer()
