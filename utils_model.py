import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):  # 修改网络结构 为duel dqn

    def __init__(self, action_dim, device):
        super(DQN, self).__init__()
        self.__conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, bias=False)
        self.__conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.__conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.__fc1 = nn.Linear(64 * 7 * 7, 512)  # 优势函数第一层fc
        self.__fc2 = nn.Linear(512, action_dim)  # 优势函数第二层fc
        self.__fc1a = nn.Linear(64 * 7 * 7, 512)  # 值函数第一层fc
        self.__fc2a = nn.Linear(512, 1)  # 值函数第二层fc
        self.__device = device

    def forward(self, x):  # 输入状态x（由连续多个frame构成的stack）
        x = x / 255.
        x = F.relu(self.__conv1(x))
        x = F.relu(self.__conv2(x))
        x = F.relu(self.__conv3(x))
        advantagex = F.relu(self.__fc1(x.view(x.size(0), -1)))
        advantage = self.__fc2(advantagex)
        valuex = F.relu(self.__fc1a(x.view(x.size(0), -1)))
        value = self.__fc2a(valuex)
        return value + (advantage - advantage.mean(1, keepdim=True))  # dueling dqn：

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
