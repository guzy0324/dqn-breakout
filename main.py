from collections import deque
import os
import random
from tqdm import tqdm

import torch

from utils_drl import Agent
from utils_env import MyEnv
from utils_memory import ReplayMemory


GAMMA = 0.99
GLOBAL_SEED = 0
MEM_SIZE = 100_000
RENDER = False
SAVE_PREFIX = "./models"
STACK_SIZE = 4

EPS_START = 1.
EPS_END = 0.1
EPS_DECAY = 1000000

BATCH_SIZE = 32
POLICY_UPDATE = 4
TARGET_UPDATE = 10_000
WARM_STEPS = 50_000
MAX_STEPS = 50_000_000
EVALUATE_FREQ = 100_000

rand = random.Random()
rand.seed(GLOBAL_SEED)
new_seed = lambda: rand.randint(0, 1000_000)
try:
    os.mkdir(SAVE_PREFIX)
except:
    pass

torch.manual_seed(new_seed())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = MyEnv(device)
agent = Agent(
    env.get_action_dim(),
    device,
    GAMMA,
    new_seed(),
    EPS_START,
    EPS_END,
    EPS_DECAY,
)
memory = ReplayMemory(STACK_SIZE + 1, MEM_SIZE, device)

#### Training ####
obs_queue: deque = deque(maxlen=5)  # 当新元素入队且队满时，会pop掉头
done = True

progressive = tqdm(range(MAX_STEPS), total=MAX_STEPS,
                   ncols=50, leave=False, unit="b")  # 进度条
for step in progressive:
    if done:  # done表示结束一次游戏，需要重置
        observations, _, _ = env.reset()
        for obs in observations:
            obs_queue.append(obs)

    training = len(memory) > WARM_STEPS
    state = env.make_state(obs_queue).to(device).float()  # 将长度5的观察队列做成state（只用到了后4个obs
    action = agent.run(state, training)
    obs, reward, done = env.step(action)
    obs_queue.append(obs)  # 将头pop，队列中剩后4个加1个新的
    memory.push(env.make_folded_state(obs_queue), action, reward, done)  # folded_state：[:4]是state，[1:]是next_state

    if step % POLICY_UPDATE == 0 and training:  # 如果training，每过POLICY_UPDATE，就更新一次policy network
        agent.learn(memory, BATCH_SIZE)

    if step % TARGET_UPDATE == 0:  # 每过TARGET_UPDATE，就更新一次target network
        agent.sync()

    if step % EVALUATE_FREQ == 0:  # 每过EVALUATE_FREQ，就评价一次
        avg_reward, frames = env.evaluate(obs_queue, agent, render=RENDER)
        with open("rewards.txt", "a") as fp:
            fp.write(f"{step//EVALUATE_FREQ:3d} {step:8d} {avg_reward:.1f}\n")
        if RENDER:  # 如果RENDER，就绘图
            prefix = f"eval_{step//EVALUATE_FREQ:03d}"
            os.mkdir(prefix)
            for ind, frame in enumerate(frames):
                with open(os.path.join(prefix, f"{ind:06d}.png"), "wb") as fp:
                    frame.save(fp, format="png")
        agent.save(os.path.join(
            SAVE_PREFIX, f"model_{step//EVALUATE_FREQ:03d}"))
        done = True
