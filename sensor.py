import global_var
import numpy as np
from Crypto.PublicKey import RSA
from Crypto.Signature import PKCS1_v1_5
from Crypto.Hash import SHA256
import random
import time


# 读取私钥
def read_private_key(file_path):
    with open(file_path, 'rb') as f:
        private_key = f.read()
    return private_key


# 读取公钥
def read_public_key(file_path):
    with open(file_path, 'rb') as f:
        public_key = f.read()
    return public_key


class Sensor(object):

    def __init__(self, Sensor_ID, sensor_params: dict, malicious_prob=0.1):
        # 节点属性
        self.Sensor_ID = Sensor_ID  # 传感节点ID
        self.type = 0  # 0为正常节点 1为故障节点，上传随机数
        self.malicious_prob = malicious_prob
        # if random.random() < self.malicious_prob:
        #     self.type = 1  # 当随机数小于概率值时，将节点类型设置为1，表示故障节点

        # 输入数据
        self.data = None
        self.data_dict = {}
        self.probability = 0.4
        # 生成密钥对
        self.private_key = None
        self.public_key = None
        # 保存传感节点信息
        # CHAIN_DATA_PATH = global_var.get_chain_data_path()
        # with open(CHAIN_DATA_PATH / f'sensor_data{str(self.Sensor_ID)}.txt', 'a') as f:
        #     print(f"Sensor {self.Sensor_ID}\n"
        #           f"private_key: {self.private_key}\n",
        #           f"public_key: {self.public_key}\n", file=f)

    # 读取指定数量的密钥对
    def read_key_pairs(self, path, path1):
        self.private_key = read_private_key(f'{path}/private_key_{self.Sensor_ID}.pem')
        self.public_key = read_public_key(f'{path1}/public_key_{self.Sensor_ID}.pem')

    def get_data(self, data):
        """
        读取仿真数据
        """
        self.data = data

    def data_process(self, timestamp, key=0):
        # timestamp = time.time()
        data_temp = np.copy(self.data)
        # if self.type == 1:
        #     data_temp[2] = random.uniform(-175, -26)
        #     data_temp = np.append(data_temp, 1)
        # else:
        #     data_temp = self.add_noise_to_gain(0, 0.5)
        #     data_temp = np.append(data_temp, 0)
        if random.random() < self.malicious_prob and key == 1:
            self.type = 1  # 当随机数小于概率值时，将节点类型设置为1，表示故障节点
            data_temp[2] = random.uniform(-175, -26)
            data_temp = np.append(data_temp, 1)
        else:
            self.type = 0
            data_temp = self.add_noise_to_gain(0, 0.5)
            data_temp = np.append(data_temp, 0)

        if global_var.get_sig():
            private_key = RSA.import_key(self.private_key)
            signer = PKCS1_v1_5.new(private_key)
            data = data_temp.tobytes()
            signature = signer.sign(SHA256.new(data))
            public_key = RSA.import_key(self.public_key)
            self.data_dict = {
                'sensing_data': data_temp,
                'timestamp': timestamp,
                'sig': signature,
                'public_key': public_key,
            }
        else:
            self.data_dict = {
                'sensing_data': data_temp,
                'timestamp': timestamp,
                'sig': None,
                'public_key': None,
            }

    def add_noise_to_gain(self, noise_mean, noise_std):
        gain = self.data[2]

        # 生成与增益数据相同大小的高斯噪声
        noise = np.random.normal(noise_mean, noise_std, gain.shape)

        # 将噪声叠加到增益数据上
        noisy_gain = gain + noise

        # 将添加噪声后的信道增益数据列替换回原数组中
        noisy_gain_array = np.array([self.data[0], self.data[1], noisy_gain])

        return noisy_gain_array

    def upload_data(self):
        # 生成一个随机概率
        prob = random.random()

        # 如果随机概率小于等于设定的上传概率，上传数据给矿工
        if prob <= self.probability:
            return self.data_dict
        else:
            return None
