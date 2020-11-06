from typing import (
    Optional,
)

import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils_types import (
    TensorStack4,
    TorchDevice,
)

from utils_memory import ReplayMemory
from utils_model import DQN


class Agent(object):

    def __init__(
            self,
            action_dim: int,
            device: TorchDevice,
            gamma: float,
            seed: int,

            eps_start: float,
            eps_final: float,
            eps_decay: float,

            restore: Optional[str] = None,
    ) -> None:
        self.__action_dim = action_dim
        self.__device = device
        self.__gamma = gamma

        self.__eps_start = eps_start
        self.__eps_final = eps_final
        self.__eps_decay = eps_decay

        self.__eps = eps_start
        self.__r = random.Random()
        self.__r.seed(seed)

        # 将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行
        self.__policy = DQN(action_dim, device).to(device)
        self.__target = DQN(action_dim, device).to(device)
        if restore is None:
            self.__policy.apply(DQN.init_weights)  # policy自定义参数初始化方式
        else:
            self.__policy.load_state_dict(
                torch.load(restore))  # policy加载之前学习到的参数
        self.__target.load_state_dict(
            self.__policy.state_dict())  # target拷贝policy的参数
        self.__optimizer = optim.Adam(  # 优化器采用Adam
            self.__policy.parameters(),
            lr=0.0000625,
            eps=1.5e-4,
        )
        self.__target.eval()  # 将模型转变为evaluation（测试）模式，这样就可以排除BN和Dropout对测试的干扰

    # epsilon-greedy
    def run(self, state: TensorStack4, training: bool = False) -> int:
        """run suggests an action for the given state."""
        if training:
            self.__eps -= \
                (self.__eps_start - self.__eps_final) / self.__eps_decay
            self.__eps = max(self.__eps, self.__eps_final)

        if self.__r.random() > self.__eps:
            with torch.no_grad():
                return self.__policy(state).max(1).indices.item()
        return self.__r.randint(0, self.__action_dim - 1)

    def learn(self, memory: ReplayMemory, batch_size: int) -> float:
        """learn trains the value network via TD-learning."""
        state_batch, action_batch, reward_batch, next_batch, done_batch = \
            memory.sample(batch_size)  # 随机取样

        values = self.__policy(state_batch.float()).gather(1, action_batch)
        values_next = self.__target(next_batch.float()).max(1).values.detach()
        expected = (self.__gamma * values_next.unsqueeze(1)) * \
            (1. - done_batch) + reward_batch
        loss = F.smooth_l1_loss(values, expected)  # smooth l1损失

        self.__optimizer.zero_grad()  # 将模型的参数梯度初始化为0
        print(loss)
        loss.backward()  # 计算梯度
        print(loss)
        for param in self.__policy.parameters():
            param.grad.data.clamp_(-1, 1)  # 固定所有参数为[-1, 1]
        self.__optimizer.step()  # 做一步最优化

        return loss.item()

    def sync(self) -> None:
        """sync synchronizes the weights from the policy network to the target
        network."""
        self.__target.load_state_dict(self.__policy.state_dict())

    def save(self, path: str) -> None:
        """save saves the state dict of the policy network."""
        torch.save(self.__policy.state_dict(), path)
