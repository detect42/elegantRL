import importlib
import os
import time
from dataclasses import dataclass, field
from multiprocessing import Pipe, Process
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf

TEN = th.Tensor


# ================================================================
# 1. 纯逻辑处理：计算衍生参数，加载类
# ================================================================
def process_config(cfg: DictConfig) -> DictConfig:
    """
    负责对 Hydra 读入的原始配置进行必要的逻辑加工。
    原则：
    1. 不拍平 (No Flattening)：保持 cfg.train.lr 这种层级。
    2. 不填默认值 (No Defaults)：Yaml 里没写的就报错。
    """
    # [Step 1] 解锁：允许写入新属性 (如 agent_class, state_dim)
    OmegaConf.set_struct(cfg, False)

    # [Step 2] 逻辑 A: 环境参数衍生计算
    # 比如 state_dim = 15 * K
    if cfg.env.get("K") is not None:
        cfg.env.state_dim = cfg.env.state_dim * cfg.env.K
        print(f"| Config: Derived state_dim={cfg.env.state_dim}")

    cfg.train.batch_size = cfg.train.batch_size * cfg.train.kk
    cfg.train.horizon_len = cfg.train.horizon_len * cfg.train.kk
    cfg.train.buffer_size = cfg.train.buffer_size * cfg.train.kk
    print(
        f"| Config: Scaled batch_size={cfg.train.batch_size}, horizon_len={cfg.train.horizon_len}, buffer_size={cfg.train.buffer_size}"
    )

    cfg.env.gpu_id = cfg.sys.gpu_id
    cfg.eval.env.gpu_id = cfg.sys.gpu_id
    cfg.eval.env.K = cfg.env.K
    # [Step 3] 逻辑 B: 自动推断 Off-policy
    if not hasattr(cfg.agent, "if_off_policy") or (cfg.agent.if_off_policy is None):
        agent_name = cfg.agent.agent_name
        on_policy_names = ("SARSA", "VPG", "A2C", "A3C", "TRPO", "PPO", "MPO")
        # 如果名字里找不到 On-Policy 的关键词，那就是 Off-Policy
        is_off_policy = all(name not in agent_name for name in on_policy_names)
        cfg.agent.if_off_policy = is_off_policy

    # [Step 4] 关锁：处理完毕，禁止后续代码随意添加新 Key，防止拼写错误
    OmegaConf.set_struct(cfg, True)

    return cfg


# ================================================================
# 2. 副作用执行：初始化环境 (单独保留)
# ================================================================
def init_before_training(cfg: DictConfig):
    """
    执行全局设置：随机种子、PyTorch 线程、CWD 记录
    """
    # 1. 随机种子
    seed = cfg.sys.random_seed
    if seed is None:
        seed = max(0, cfg.sys.gpu_id)  # 默认用 GPU ID

        # 因为我们之前关锁了，这里临时解锁修改一下 seed
        OmegaConf.set_struct(cfg, False)
        cfg.sys.random_seed = seed
        OmegaConf.set_struct(cfg, True)

    np.random.seed(seed)
    th.manual_seed(seed)

    # 2. PyTorch 线程
    th.set_num_threads(cfg.sys.num_threads)
    th.set_default_dtype(th.float32)

    # 3. 记录 CWD (Current Working Directory)
    # Hydra 已经切换了目录，这里记录一下供 Agent 保存模型用
    current_log_dir = os.getcwd()

    # 临时解锁写入 cwd
    OmegaConf.set_struct(cfg, False)
    cfg.eval.cwd = current_log_dir
    OmegaConf.set_struct(cfg, True)

    print(f"| Init: Seed={seed}, Threads={cfg.sys.num_threads}")
    print(f"| Init: CWD={cfg.eval.cwd}")


# ================================================================
# 辅助工具
# ================================================================
def get_class_from_path(path: str):
    """根据字符串路径加载类"""
    try:
        if "." in path:
            module_path, class_name = path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        else:
            # 默认尝试从 elegantrl.agents 找
            import elegantrl.agents

            return getattr(elegantrl.agents, path)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot import class: {path}. {e}")


def build_env(env_class=None, cfg: Optional[DictConfig] = None, gpu_id: int = -1):
    import warnings

    warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated.*")
    # env_args["gpu_id"] = gpu_id  # set gpu_id for vectorized env before build it

    if cfg.if_build_vec_env:
        env = VecEnv(env_class=env_class, cfg=cfg, gpu_id=gpu_id)
    elif env_class.__module__ == "gymnasium.envs.registration":
        env = env_class(id=cfg.env_name)
    else:
        env = env_class(cfg)

    for attr_str in ("env_name", "num_envs", "max_step", "state_dim", "action_dim", "if_discrete"):
        assert hasattr(env, attr_str), f"Environment missing required attribute: {attr_str}"
    return env


def kwargs_filter(function, kwargs: dict) -> dict:
    import inspect

    sign = inspect.signature(function).parameters.values()
    sign = {val.name for val in sign}
    common_args = sign.intersection(kwargs.keys())
    return {key: kwargs[key] for key in common_args}  # filtered kwargs


def get_gym_env_args(env, if_print: bool) -> dict:
    """get a dict about a standard OpenAI gym env information.

    param env: a standard OpenAI gym env
    param if_print: [bool] print the dict about env information.
    return: env_args [dict]

    env_args = {
        'env_name': env_name,       # [str] the environment name, such as XxxXxx-v0
        'num_envs': num_envs.       # [int] the number of sub envs in vectorized env. `num_envs=1` in single env.
        'max_step': max_step,       # [int] the max step number of an episode.
        'state_dim': state_dim,     # [int] the dimension of state
        'action_dim': action_dim,   # [int] the dimension of action or the number of discrete action
        'if_discrete': if_discrete, # [bool] action space is discrete or continuous
    }
    """
    import warnings

    warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated.*")

    if_gym_standard_env = {"unwrapped", "observation_space", "action_space", "spec"}.issubset(dir(env))

    if if_gym_standard_env and (not hasattr(env, "num_envs")):  # isinstance(env, gym.Env):
        env_name = env.unwrapped.spec.id
        num_envs = getattr(env, "num_envs", 1)
        max_step = getattr(env, "_max_episode_steps", 12345)

        state_shape = env.observation_space.shape
        state_dim = state_shape[0] if len(state_shape) == 1 else state_shape  # sometimes state_dim is a list

        if_discrete = str(env.action_space).find("Discrete") >= 0
        if if_discrete:  # make sure it is discrete action space
            action_dim = getattr(env.action_space, "n")
        elif str(env.action_space).find("Box") >= 0:  # make sure it is continuous action space
            action_dim = env.action_space.shape[0]
            if any(env.action_space.high - 1):
                print(f"| WARNING: env.action_space.high {env.action_space.high}", flush=True)
            if any(env.action_space.low + 1):
                print(f"| WARNING: env.action_space.low {env.action_space.low}", flush=True)
        else:
            raise RuntimeError(
                "\n| Error in get_gym_env_info(). Please set these value manually:"
                "\n  `state_dim=int; action_dim=int; if_discrete=bool;`"
                "\n  And keep action_space in range (-1, 1)."
            )
    else:
        env_name = getattr(env, "env_name", "env")
        num_envs = getattr(env, "num_envs", 1)
        max_step = getattr(env, "max_step", 12345)
        state_dim = env.state_dim
        action_dim = env.action_dim
        if_discrete = env.if_discrete

    env_args = {
        "env_name": env_name,
        "num_envs": num_envs,
        "max_step": max_step,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "if_discrete": if_discrete,
    }
    if if_print:
        env_args_str = repr(env_args).replace(",", f",\n{'':11}")
        print(f"env_args = {env_args_str}", flush=True)
    cfg = DictConfig(env_args)
    return cfg


"""vectorized env"""


class SubEnv(Process):
    def __init__(self, sub_pipe0: Pipe, vec_pipe1: Pipe, env_class, cfg: DictConfig, env_id: int = 0):
        super().__init__()
        self.sub_pipe0 = sub_pipe0
        self.vec_pipe1 = vec_pipe1

        self.env_class = env_class
        self.cfg = cfg
        self.env_id = env_id

    def run(self):
        th.set_grad_enabled(False)

        """build env"""
        if self.env_class.__module__ == "gymnasium.envs.registration":  # is standard OpenAI Gym env
            env = self.env_class(id=self.cfg.env_name)
        else:
            env = self.env_class(self.cfg)

        """set env random seed"""
        random_seed = self.env_id
        np.random.seed(random_seed)
        th.manual_seed(random_seed)

        while True:
            action = self.sub_pipe0.recv()
            if action is None:
                state, info_dict = env.reset()
                self.vec_pipe1.send((self.env_id, state))
            else:
                state, reward, terminal, truncate, info_dict = env.step(action)

                done = terminal or truncate
                state = env.reset()[0] if done else state
                self.vec_pipe1.send((self.env_id, state, reward, terminal, truncate))


class VecEnv:
    def __init__(self, env_class: object, cfg: DictConfig, gpu_id: int = -1):
        self.device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.num_envs = cfg.num_envs  # the number of sub env in vectorized env.
        assert self.num_envs > 1
        """the necessary env information when you design a custom env"""
        self.env_name = cfg.env_name  # the name of this env.
        self.max_step = cfg.max_step  # the max step number in an episode for evaluation
        self.state_dim = cfg.state_dim  # feature number of state
        self.action_dim = cfg.action_dim  # feature number of action
        self.if_discrete = cfg.if_discrete  # discrete action or continuous action
        """speed up with multiprocessing: Process, Pipe"""
        assert self.num_envs <= 64
        self.res_list = [[] for _ in range(self.num_envs)]

        sub_pipe0s, sub_pipe1s = list(zip(*[Pipe(duplex=False) for _ in range(self.num_envs)]))
        self.sub_pipe1s = sub_pipe1s

        vec_pipe0, vec_pipe1 = Pipe(duplex=False)  # recv, send
        self.vec_pipe0 = vec_pipe0

        self.sub_envs = [
            SubEnv(sub_pipe0=sub_pipe0, vec_pipe1=vec_pipe1, env_class=env_class, cfg=cfg, env_id=env_id)
            for env_id, sub_pipe0 in enumerate(sub_pipe0s)
        ]

        [setattr(p, "daemon", True) for p in self.sub_envs]  # set before process start to exit safely
        [p.start() for p in self.sub_envs]

    def reset(self) -> Tuple[TEN, dict]:  # reset the agent in env
        th.set_grad_enabled(False)

        for pipe in self.sub_pipe1s:
            pipe.send(None)
        (states,) = self.get_orderly_zip_list_return()
        states = th.tensor(np.stack(states), dtype=th.float32, device=self.device)
        info_dicts = dict()
        return states, info_dicts

    def step(self, action: TEN) -> Tuple[TEN, TEN, TEN, TEN, dict]:  # agent interacts in env
        action = action.detach().cpu().numpy()
        for pipe, a in zip(self.sub_pipe1s, action):
            pipe.send(a)

        states, rewards, terminal, truncate = self.get_orderly_zip_list_return()
        states = th.tensor(np.stack(states), dtype=th.float32, device=self.device)
        rewards = th.tensor(rewards, dtype=th.float32, device=self.device)
        terminal = th.tensor(terminal, dtype=th.bool, device=self.device)
        truncate = th.tensor(truncate, dtype=th.bool, device=self.device)
        info_dicts = dict()
        return states, rewards, terminal, truncate, info_dicts

    def close(self):
        [process.terminate() for process in self.sub_envs]

    def get_orderly_zip_list_return(self):
        for _ in range(self.num_envs):
            res = self.vec_pipe0.recv()
            self.res_list[res[0]] = res[1:]
        return list(zip(*self.res_list))


def check_vec_env():
    import gymnasium as gym

    num_envs = 3
    gpu_id = 0

    env_class = gym.make  # run a custom env: PendulumEnv, which based on OpenAI pendulum
    env_args = {
        "env_name": "CartPole-v1",
        "max_step": 500,
        "state_dim": 4,
        "action_dim": 2,
        "if_discrete": True,
    }
    cfg = DictConfig(env_args)

    env = VecEnv(env_class=env_class, cfg=cfg, num_envs=num_envs, gpu_id=gpu_id)

    device = th.device(f"cuda:{gpu_id}" if (th.cuda.is_available() and (gpu_id >= 0)) else "cpu")
    state, info_dict = env.reset()
    print(f"| num_envs {num_envs}  state.shape {state.shape}", flush=True)

    for i in range(4):
        if env.if_discrete:  # state -> action
            action = th.randint(0, env.action_dim, size=(num_envs,), device=device)
        else:
            action = th.zeros(size=(num_envs,), dtype=th.float32, device=device)
        state, reward, terminal, truncate, info_dict = env.step(action)

        print(f"| num_envs {num_envs}  {[t.shape for t in (state, reward, terminal, truncate)]}", flush=True)
    env.close() if hasattr(env, "close") else None


if __name__ == "__main__":
    check_vec_env()
