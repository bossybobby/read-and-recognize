import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    """简单的CNN用于手写数字识别"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class DigitRecognizer:
    def __init__(self):
        self.model = SimpleCNN()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式

        # 加载预训练权重（如果有的话）
        # self.load_pretrained_model()

    def preprocess_capacitive_data(self, capacitive_data):
        """
        预处理电容数据为适合CNN的格式
        输入: 8x8 numpy array
        输出: 1x28x28 tensor (模拟MNIST格式)
        """
        if capacitive_data.shape != (8, 8):
            raise ValueError("输入数据必须是8x8")

        # 归一化到0-1
        data_min = capacitive_data.min()
        data_max = capacitive_data.max()
        if data_max > data_min:
            normalized = (capacitive_data - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(capacitive_data)

        # 插值到28x28 (使用简单的最近邻插值)
        from scipy import ndimage
        resized = ndimage.zoom(normalized, 28/8, order=0)

        # 转换为tensor并添加批次和通道维度
        tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, capacitive_data):
        """
        预测手写数字
        输入: 8x8 numpy array
        输出: (预测数字, 置信度)
        """
        try:
            processed_data = self.preprocess_capacitive_data(capacitive_data)
            with torch.no_grad():
                output = self.model(processed_data)
                probabilities = torch.exp(output)
                confidence, predicted = torch.max(probabilities, 1)
                return predicted.item(), confidence.item()
        except Exception as e:
            print(f"预测错误: {e}")
            return -1, 0.0

    def load_pretrained_model(self, model_path='mnist_cnn.pth'):
        """加载预训练模型"""
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("成功加载预训练模型")
        except FileNotFoundError:
            print("未找到预训练模型，使用随机初始化权重")
        except Exception as e:
            print(f"加载模型失败: {e}")

def train_mnist_model(save_path='mnist_cnn.pth'):
    """训练MNIST模型（可选，用于生成预训练权重）"""
    # 这里可以添加MNIST训练代码
    # 由于需要下载数据集，这里先跳过
    print("训练功能暂未实现，请使用预训练模型")

def simulate_digit_patterns():
    """生成模拟的手写数字模式用于测试"""
    patterns = {
        0: np.array([
            [0,1,1,1,0],
            [1,0,0,0,1],
            [1,0,0,0,1],
            [1,0,0,0,1],
            [0,1,1,1,0]
        ]),
        1: np.array([
            [0,0,1,0,0],
            [0,1,1,0,0],
            [0,0,1,0,0],
            [0,0,1,0,0],
            [0,1,1,1,0]
        ]),
        2: np.array([
            [0,1,1,1,0],
            [1,0,0,0,1],
            [0,0,1,1,0],
            [0,1,0,0,0],
            [1,1,1,1,1]
        ]),
        # 可以添加更多数字...
    }

    # 扩展到8x8
    expanded_patterns = {}
    for digit, pattern in patterns.items():
        expanded = np.zeros((8, 8))
        start = (8 - 5) // 2
        expanded[start:start+5, start:start+5] = pattern
        # 缩放值到电容范围
        expanded_patterns[digit] = (expanded * 600 + 200).astype(int)

    return expanded_patterns

if __name__ == "__main__":
    # 测试识别器
    recognizer = DigitRecognizer()

    # 使用模拟模式
    patterns = simulate_digit_patterns()

    print("测试数字识别:")
    for digit, pattern in patterns.items():
        pred, conf = recognizer.predict(pattern)
        print(f"真实数字: {digit}, 预测: {pred}, 置信度: {conf:.3f}")

        # 显示模式
        plt.figure(figsize=(4, 4))
        plt.imshow(pattern, cmap='viridis')
        plt.title(f"数字 {digit} - 预测 {pred} (置信度: {conf:.3f})")
        plt.colorbar()
        plt.show()