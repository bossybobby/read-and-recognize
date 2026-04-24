import os
import pandas as pd
import numpy as np
from pathlib import Path

class CapacitiveDataReader:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)

    def get_latest_csv(self):
        """获取最新的CSV文件"""
        csv_files = list(self.log_dir.glob("Data_*.csv"))
        if not csv_files:
            return None
        return max(csv_files, key=lambda x: x.stat().st_mtime)

    def parse_csv_data(self, csv_path):
        """解析CSV文件中的电容数据"""
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 解析头部信息
        header = {}
        data_line = None
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                header[key.strip()] = value.strip()
            elif line and not line.startswith('$'):
                data_line = line
                break

        # 解析数据
        if data_line:
            values = [int(x.strip()) for x in data_line.split(',') if x.strip()]
            tx_num = int(header.get('TxNum', 100))
            rx_num = int(header.get('RxNum', 175))

            # 重塑为二维数组
            if len(values) == tx_num * rx_num:
                data = np.array(values).reshape(tx_num, rx_num)
                return header, data

        return header, None

    def get_simulated_data(self):
        """生成模拟的8x8电容数据用于测试"""
        # 模拟一个8x8的电容阵列，值在100-1000之间
        np.random.seed(42)  # 固定种子以便重现
        base_values = np.random.randint(200, 800, (8, 8))
        # 添加一些噪声
        noise = np.random.normal(0, 50, (8, 8))
        simulated_data = base_values + noise
        return np.clip(simulated_data, 0, 1000).astype(int)

if __name__ == "__main__":
    # 示例用法 - 模拟数据
    reader = CapacitiveDataReader("dummy")  # 不需要真实路径

    print("使用模拟数据:")
    simulated_data = reader.get_simulated_data()
    print(f"模拟数据形状: {simulated_data.shape}")
    print("模拟8x8数据:")
    print(simulated_data)

    # 如果有真实数据，也可以读取
    # log_dir = r"../istaricMDT-V20241130/istaricMDT-V20241130/User/Data/Log"
    # reader = CapacitiveDataReader(log_dir)
    # latest_csv = reader.get_latest_csv()
    # if latest_csv:
    #     print(f"读取文件: {latest_csv}")
    #     header, data = reader.parse_csv_data(latest_csv)
    #     print(f"帧ID: {header.get('FrameId')}")
    #     print(f"阵列尺寸: {header.get('TxNum')}x{header.get('RxNum')}")
    #
    #     if data is not None:
    #         print(f"数据形状: {data.shape}")
    #         print("前8x8数据:")
    #         sub_data = reader.get_8x8_data(data)
    #         if sub_data is not None:
    #             print(sub_data)
    # else:
    #     print("未找到CSV文件，使用模拟数据")