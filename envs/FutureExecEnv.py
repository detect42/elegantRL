import json
import os
import random
import time
from typing import List, Optional, Tuple

import numpy as np
import numpy.random as rd
import pandas as pd
import torch as th
from .auxiliary_data import error_idx
from omegaconf import DictConfig, OmegaConf

ARY = np.ndarray
np.random.seed(527)
# random.seed(527)


class FutureExecEnv:

    def __init__(
        self,
        cfg: DictConfig,
    ):
        self.num_envs = cfg.num_envs
        self.K: int = cfg.K
        self.state_dim: int = cfg.state_dim
        self.action_dim: int = cfg.action_dim
        self.if_discrete: bool = cfg.if_discrete
        self.padding0: bool = cfg.padding0
        self.max_step: int = cfg.max_step
        self.eval_num_workers: int = cfg.eval_num_workers
        self.device = th.device("cpu") if cfg.gpu_id == -1 else th.device(f"cuda:{cfg.gpu_id}")
        self.tot_uncompleted = 0
        self.dataset = "1_sample"
        self.position_root = "/nas/srwang/ExecRL_DATA/1_sample_signals_norm"
        self.scored_tick2_root = "/nas/srwang/ExecRL_DATA/1_sample_signals_norm"
        self.position: Optional[pd.DataFrame] = None
        self.tick2: Optional[pd.DataFrame] = None
        self.action_range = cfg.action_range if self.if_discrete else []
        # reset()

        self.id_index = 0
        self.idx_index_within = 0
        self.id: str = ""
        self.uncompleted = False  # whether the current episode is uncompleted
        # environment information
        self.env_name: str = "FutureExecEnv_v8"

        # self.max_step = 19260817
        self.freq = 5
        self.time_curr: int = 0
        self.realized_position: int = 0
        self.from_position: int = 0
        self.to_position: int = 0
        self.side: int = 0
        self.time_idx: int = 0
        self.contract_multiplier: int = 1
        self.total_time: int = 0
        self.cum_volume: float = 0.0
        self.cum_turnover: float = 0.0
        self.cum_slippage: float = 0.0
        self.total_turnover: float = 0.0
        self.base_price: float = 0.0
        self.history: list = []

        # 加载 json 数据
        print(f"/code/srwang/Finrl/sample_pool/1_train_data.json")
        with open(f"/code/srwang/Finrl/sample_pool/1_train_data.json", "r") as f:
            sample_data = json.load(f)
        print("sample data len= ", len(sample_data))
        self.error_idx = error_idx
        self.sample_pool = sample_data[:]
        self.sample_pool_size = len(self.sample_pool)
        eval_data = json.load(open(f"/code/srwang/Finrl/sample_pool/1_eval_data.json", "r"))
        self.eval_pool = eval_data[:50000]
        self.eval_pool_size = len(self.eval_pool)
        self.begin_time = 0
        self.end_time = 0
        self.cum_reward: float = 0.0

    def sample(self, sequential=False):
        if sequential:
            # 如果是第一次顺序采样，初始化索引
            if not hasattr(self, "_sequential_index"):
                self._sequential_index = 0

            # 顺序采样，可能需要跳过某些样本
            while True:
                sample = self.sample_pool[self._sequential_index]
                self._sequential_index = (self._sequential_index + 1) % len(self.sample_pool)

                id = sample["id"]
                if id in self.error_idx:
                    continue

                return sample
        else:
            # 原有的随机采样逻辑
            while True:
                sample = random.choice(self.sample_pool)
                id = sample["id"]
                if id in self.error_idx:
                    continue

                return sample

    def reset(self, set_id: int = -1, sequential=False, eval_mode=False) -> Tuple[ARY, dict]:
        if set_id == -1:
            sample = self.sample(sequential)
        else:
            if eval_mode:
                sample = self.eval_pool[set_id]
            else:
                sample = self.sample_pool[set_id]
        self.samples = sample
        self.id = sample["id"]
        self.begin_time = sample["begin_time"]
        self.end_time = sample["end_time"]

        self.realized_position = 0
        self.cum_volume = 0.0
        self.cum_turnover = 0.0
        self.cum_slippage = 0.0
        self.total_turnover = 0.0
        self.from_position = 0
        self.to_position = sample["target_position"]
        self.total_time = self.end_time - self.begin_time
        self.uncompleted = False
        self.cum_reward = 0.0
        assert self.state_dim % self.K == 0, f"state_dim={self.state_dim} must be divisible by K={self.K}"
        self.history.clear()
        # 构造 position DataFrame
        times = np.arange(self.begin_time, self.end_time + 1, self.freq)  #! 注意data里面 结尾没有按照5s对齐
        self.position = pd.DataFrame({"version_ts": times, "position": self.to_position}, index=times)

        self.tick2 = pd.read_pickle(os.path.join(self.scored_tick2_root, f"tick2_{self.id}.pkl"))
        self.tick2["version_ts"] = self.tick2.index
        self.tick2 = self.tick2[
            (self.tick2["version_ts"] >= self.begin_time) & (self.tick2["version_ts"] <= self.end_time)
        ]
        self.process()
        self.base_price = self.position.iloc[0]["mid_price"]
        self.total_turnover = self.base_price * np.abs(
            self.to_position
        )  #! 之前没加abs,导致在negative情况下全是负的total_turnover
        assert self.from_position != self.to_position, "from_position should not equal to to_position at reset."
        self.side = np.sign(self.to_position - self.from_position)
        # print(self.from_position, "-->",self.to_position, self.side)
        self.time_idx = 0
        self.time_curr = self.position.index[self.time_idx]
        # print(self.tick2)
        # print(self.position)

        new_state = self.get_state()
        self.history.append(new_state)
        if len(self.history) > self.K:
            self.history.pop(0)
        while len(self.history) < self.K:
            if self.padding0:
                self.history.insert(0, np.zeros_like(self.history[0]))
            else:
                self.history.insert(0, self.history[0])

            # self.history.insert(0, np.zeros_like(self.history[0]))
        assert len(self.history) == self.K
        state_stack = np.concatenate(self.history, axis=0)
        return state_stack, {
            "id": self.id,
            "begin_time": self.begin_time,
            "target_position": self.to_position,
            "end_time": self.end_time,
        }

    def process(self):

        assert not ((self.tick2["ask_price1"] <= 0) & (self.tick2["bid_price1"] <= 0)).any()
        self.tick2["mid_price"] = (self.tick2["ask_price1"] + self.tick2["bid_price1"]) / 2
        self.tick2.loc[self.tick2["ask_price1"] <= 0, "mid_price"] = self.tick2.loc[
            self.tick2["ask_price1"] <= 0, "bid_price1"
        ]
        self.tick2.loc[self.tick2["bid_price1"] <= 0, "mid_price"] = self.tick2.loc[
            self.tick2["bid_price1"] <= 0, "ask_price1"
        ]

        idxs = np.searchsorted(self.tick2["version_ts"].values, self.position["version_ts"].values)
        idxs = np.clip(idxs, 0, len(self.tick2) - 1)
        self.position["mid_price"] = self.tick2.iloc[idxs]["mid_price"].values
        self.position["volume"] = self.tick2.iloc[idxs]["volume"].values
        self.position["turnover"] = self.tick2.iloc[idxs]["turnover"].values
        self.position["volume"] = self.position["volume"].diff().fillna(0.0)
        self.position["turnover"] = self.position["turnover"].diff().fillna(0.0)
        self.position["vwap"] = self.position["turnover"] / self.position["volume"] / self.contract_multiplier
        self.position["last_price"] = self.tick2.iloc[idxs]["last_price"].values
        valid_mask = (0.8 * self.position["mid_price"] <= self.position["vwap"]) & (
            self.position["vwap"] <= 1.2 * self.position["mid_price"]
        )
        self.position.loc[~valid_mask, "vwap"] = self.position.loc[~valid_mask, "mid_price"]
        self.position["dh"] = np.nan
        self.position["tick2_idx"] = idxs

        self.position["realized_position"] = 0
        self.position["yeday_real"] = self.position["realized_position"].copy()

        # Precompute rolling features in tick2 for get_state
        idxs = np.searchsorted(self.tick2["version_ts"].values, self.position["version_ts"].values)
        idxs = np.clip(idxs, 0, len(self.tick2) - 1)
        feature_cols = ["signal_short", "signal_middle", "signal_long_1h"]
        tick_features = self.tick2.loc[:, feature_cols].iloc[idxs].reset_index(drop=True)
        self.position.loc[:, feature_cols] = tick_features.values

    def calc_expected_slippage(
        self, tick2: pd.DataFrame, delta_position: int, base_price: float, side: float, calc_realized_vol: bool = False
    ) -> float:
        """
        Calculate the expected slippage based on the tick2 data and the delta_position.
        This function assumes that tick2 contains the necessary price and volume information.
        """
        if delta_position == 0:
            return 0.0
        sign = np.sign(delta_position)
        abs_delta = abs(delta_position)
        total_volume = 0.0
        total_slippage = 0.0

        for i in range(1, 6):
            volume_col = f"{'ask' if sign == 1 else 'bid'}_volume{i}"
            price_col = f"{'ask' if sign == 1 else 'bid'}_price{i}"
            if i < 5:
                volume = min(tick2[volume_col], abs_delta - total_volume)
                price = tick2[price_col]
            else:
                volume = abs_delta
                price = tick2[f"{'ask' if sign == 1 else 'bid'}_price5"] + sign * 1
            abs_delta -= volume
            if calc_realized_vol:
                self.realized_position += volume * sign
            total_slippage += volume * (price - base_price) * side
            if abs_delta == 0:
                break

        # print("total slippage:",total_slippage)
        return total_slippage

    @staticmethod
    def trans_dict(state_dict):

        # state_dict["realized_position_ratio"] = state_dict["realized_position_ratio"] / 8
        state_dict["signal_short"] = state_dict["signal_short"] * state_dict["side"]
        state_dict["signal_middle"] = state_dict["signal_middle"] * state_dict["side"]
        state_dict["signal_long_1h"] = state_dict["signal_long_1h"] * state_dict["side"]
        state_dict["cur_price/base_price"] = (
            np.sign(state_dict["cur_price/base_price"])
            * np.log(1 + np.abs(state_dict["cur_price/base_price"]) / 3)
            * state_dict["side"]
        )
        state_dict["exp_slippage_bp"] = np.sign(state_dict["exp_slippage_bp"]) * np.log(
            1 + np.abs(state_dict["exp_slippage_bp"]) / 3
        )
        state_dict["now_slippage_bp"] = np.sign(state_dict["now_slippage_bp"]) * np.log(
            1 + np.abs(state_dict["now_slippage_bp"]) / 3
        )
        state_dict.pop("side", None)
        return state_dict

    def get_state(self) -> ARY:
        if self.position is None or self.tick2 is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        pos = self.position.loc[self.time_curr]
        consumed_time = self.time_curr - self.begin_time
        remain_time = self.end_time - self.time_curr
        remain_pos_abs = np.abs(self.to_position - self.realized_position)
        """print(
            self.realized_position,
            "-->",
            self.to_position,
            "\n",
            self.total_time,
            " consumed_time",
           consumed_time,
            " volume_30s:",
            pos["vol_15s"],
        )"""
        state_dict: dict[str, float] = {}
        state_dict["side"] = self.side  # 方向
        state_dict["log_position"] = np.log(1 + np.abs(self.to_position) / 50)  # 方向
        state_dict["realized_ratio"] = self.realized_position / self.to_position  # 已实现仓位占目标仓位的比例
        state_dict["consumed_time_ratio"] = consumed_time / self.total_time  # 已消耗时间占总时间的比例
        if remain_time == 0:
            print(self.id, self.begin_time, self.end_time)
            raise ValueError("No remaining time in the episode.")
        state_dict["realized_position_ratio"] = np.abs(self.realized_position) / ((self.cum_volume / 100) + 1e-4)
        state_dict["signal_short"] = pos["signal_short"]  # if pos["signal_short"] is not None else 0.0
        state_dict["signal_middle"] = pos["signal_middle"]  # if pos["signal_middle"] is not None else 0.0
        state_dict["signal_long_1h"] = 0.0 if pd.isna(pos["signal_long_1h"]) else pos["signal_long_1h"]

        state_dict["cur_price/base_price"] = (
            pos["last_price"] / self.base_price - 1
        ) * 10000  # 当前价格与基准价格的比率
        tick_time = int(pos["tick2_idx"])
        state_dict["exp_slippage_bp"] = (
            (
                self.calc_expected_slippage(
                    self.tick2.iloc[tick_time], self.to_position - self.realized_position, self.base_price, self.side
                )
                + self.cum_slippage
            )
            / self.total_turnover
            * 10000
        )  # 预期滑点，单位为基点
        state_dict["now_slippage_bp"] = (self.cum_slippage) / (self.cum_turnover + 1) * 10000  # 当前滑点，单位为基点

        state_dict_trans = self.trans_dict(state_dict)
        # print(state_dict_trans)  # pr
        # 拼接到 State 后面
        # print(state_dict)
        State = np.array(list(state_dict_trans.values()), dtype=np.float32)
        # print(State)
        # time.sleep(1)
        if random.random() < 0.0000002:
            print("state", State.shape, [f"{x:.3f}" for x in State])

        # print(State.shape, "State shape")
        # print(State)
        # print("State:", State)
        if not np.isfinite(State).all():
            print("Warning: Non-finite values detected in state features, replacing with zeros.")
            print(State)
            print(state_dict)
            print(self.samples)
            State = np.nan_to_num(State, nan=0.0, posinf=0.0, neginf=0.0)
            raise ValueError("Non-finite values detected in state features.")
        return State

    def Calc_reward(self, delta_p, base_price, vwap) -> float:
        # print("delta_p:", delta_p, "mid_price:", mid_price, "vwap:", vwap)
        penalty = 0.0
        slippage = delta_p * (vwap - base_price)
        reward = -slippage
        reward -= penalty
        return reward

    @staticmethod
    def convert_action_for_env(action):
        # return action
        return 0.04 * (action + 1)

    def step(self, action) -> Tuple[ARY, float, bool, bool, dict]:
        if self.position is None or self.tick2 is None:
            raise ValueError("Environment not initialized. Call reset() first.")
        state = self.get_state()
        terminated = False
        truncated = False
        i = self.time_idx
        cur_ts = self.position.index[i]
        pre_ts = cur_ts - 120
        cur_rate = self.action_range[int(action)] if self.if_discrete else self.convert_action_for_env(action)
        # if random.random() < 0.00005:
        #    print("action:", cur_rate)

        # print(cur_rate, "cur_rate") #pr
        # print(self.realized_position, "realized_position")
        # print(f"Step {i}, cur_ts: {cur_ts}, cur_rate: {cur_rate}")
        dh = self.to_position - self.realized_position
        side = np.sign(dh)
        dh = abs(dh)
        self.cum_volume += self.position.iloc[i + 1]["volume"]
        cap = cur_rate * self.position.iloc[i + 1]["volume"]
        cap_int = int(cap)
        cap_xs = cap - cap_int
        num = np.random.random()
        if num < cap_xs:
            cap_int += 1
        cap = cap_int
        dh = min(dh, cap)

        cur_idx = self.position.index[i]
        new_idx = self.position.index[i + 1]
        self.realized_position = int(self.realized_position + side * dh)
        """print(
            self.from_position,
            "-->",
            self.to_position,
            "side:",
            side,
            "dh:",
            dh,
            "realized_position:",
            self.realized_position,
        )"""
        self.time_idx += 1
        self.time_curr += self.freq
        self.cum_turnover += dh * self.position.iloc[i + 1]["vwap"]
        self.cum_slippage += dh * side * (self.position.iloc[self.time_idx]["vwap"] - self.base_price)
        # print(self.cum_slippage, "cum_slippage")
        reward = self.Calc_reward(  #! 这里必须要带方向的变化量
            dh * side, self.base_price, self.position.iloc[self.time_idx]["vwap"]
        )
        self.cum_reward += reward
        if self.time_idx == len(self.position) - 1:
            terminated = True
            if self.realized_position != self.to_position:
                self.uncompleted = True

        penalty = 0
        reward -= penalty

        while self.realized_position == self.to_position and not terminated:
            self.time_idx += 1
            self.time_curr += self.freq
            if self.time_idx == len(self.position) - 1:
                terminated = True
                break
        # print(reward, "reward")
        if self.time_idx == len(self.position) - 1:
            # print("!")
            if self.uncompleted:
                self.tot_uncompleted += 1
                final_slippage = (
                    self.calc_expected_slippage(
                        self.tick2.iloc[-1],
                        self.to_position - self.realized_position,
                        self.base_price,
                        self.side,
                        calc_realized_vol=True,
                    )
                    * 1.0
                )  # pe
                self.cum_reward -= final_slippage
                reward -= final_slippage

            assert self.realized_position == self.to_position, "Realized position does not match target position."

        if terminated:
            new_state = state
        else:
            new_state = self.get_state()

        self.history.append(new_state)
        if len(self.history) > self.K:
            self.history.pop(0)
        while len(self.history) < self.K:
            if self.padding0:
                self.history.insert(0, np.zeros_like(self.history[0]))
            else:
                self.history.insert(0, self.history[0])

        assert len(self.history) == self.K
        # print(self.history)
        state_stack = np.concatenate(self.history, axis=0)
        return state_stack, reward / 1, terminated, truncated, {}


def check_stock_trading_env():
    import numpy as np
    import torch

    random.seed(527)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_path_positive = "/code/srwang/Finrl/log/20250830-235201_modsac-discrete v7 upper=0.08 TCN+TCN[1 2 3]  R.d.1  emb_ch=80  negative buffer=1e6 [0 2 4 6 8]/actor__000078348288.pt"
    actor_path_negative = "/code/srwang/Finrl/log/20250830-235201_modsac-discrete v7 upper=0.08 TCN+TCN[1 2 3]  R.d.1  emb_ch=80  negative buffer=1e6 [0 2 4 6 8]/actor__000078348288.pt"
    device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    actor_positive = torch.load(actor_path_positive, map_location=device, weights_only=False)
    actor_negative = torch.load(actor_path_negative, map_location=device, weights_only=False)
    env = FutureExecEnv(position_sign="negative", K=13)
    env.if_discrete = True
    """state, info = env.reset(
        set_id={
            "id": "500006676",
            "begin_time": 60.0,
            "target_position": -30.0,
            "end_time": 795.0,
            "weight": 1.0,
            "type": "short",
            "class": "plus5_sample",
        }
    )  # Example ID and index"""
    state, info = env.reset(sequential=True)
    # print("Initial State:", state)
    slippage_bp = []
    action = 1
    rrr = [0.00, 0.02, 0.04, 0.06, 0.08]
    reward_cum = 0
    id = 0
    type_position = 2
    if type_position == 1:
        actor = actor_positive
    elif type_position == 2:
        actor = actor_negative
    from tqdm import tqdm

    for _ in tqdm(range(500000000), desc="Simulation Progress"):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # print(state.shape)
        # action = actor(state).detach().cpu().item()
        action = 1
        # print(action)
        # action = 0
        if _ <= 300:
            # print("state: ", state)
            print("Action:", rrr[action])
        state, reward, terminated, truncated, info = env.step(action)
        # print("State:", state)
        """print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)"""
        # print("Info:", info)
        reward_cum += reward
        # print("Reward: ", reward)
        if terminated or truncated:
            slippage_bp.append(-reward_cum)
            if random.random() < 0.0005:
                print(-reward_cum, np.mean(slippage_bp))
            # print("-----------end-----------")
            id += 1
            # 173281
            # 38539
            # 43912
            # 347709 - 10 = 347699
            if id == 347699:
                break
            state, info = env.reset(sequential=True)
            # print(info)
            reward_cum = 0

    print("slippage value=", slippage_bp)
    print(np.mean(slippage_bp))
    print(env.tot_uncompleted)


if __name__ == "__main__":
    check_stock_trading_env()
