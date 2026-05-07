import json
import os
import random
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import torch as th

# 全局随机种子
np.random.seed(527)
random.seed(527)
ARY = np.ndarray

import os
import random
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from typing import List, Tuple, Dict, Any

from sample_pool import SamplePool


class AStockExecutionEnv:
    def __init__(self, cfg: DictConfig):
        # 1. 配置参数提取 (严格访问，缺失即报错)
        self.num_envs = cfg.num_envs
        self.K: int = cfg.K
        self.state_dim: int = cfg.state_dim
        assert self.state_dim % self.K == 0, f"state_dim={self.state_dim} must be divisible by K={self.K}"
        self.action_dim: int = cfg.action_dim
        self.lambda_decay: float = cfg.lambda_decay
        self.sample_method: str = cfg.sample_method
        self.if_discrete: bool = cfg.if_discrete
        self.padding0: bool = cfg.padding0
        self.max_step: int = cfg.max_step
        self.eval_num_workers: int = cfg.eval_num_workers
        self.noise_std_ratio: float = cfg.noise_std_ratio
        self.device = th.device("cpu") if cfg.gpu_id == -1 else th.device(f"cuda:{cfg.gpu_id}")
        self.mode:str = cfg.mode 
        self.env_name: str = "AStockExecutionEnv"
        print(f"Initialized AStockExecutionEnv with mode={self.mode}, num_envs={self.num_envs}, K={self.K}, state_dim={self.state_dim}, action_dim={self.action_dim}, if_discrete={self.if_discrete})")
        # 2. 状态特征定义 (静态 + 动态)
        # fast_signal_20260310_v1_norm,buy_1m_ret_norm,sell_1m_ret_norm,pv_corr_40x3s,KER_40x3s,Vol_Squeeze,
        # #vol_shock,vol_feat_300s,smart_momentum_300s,Gini_300s,pressure_60s,code,cmf_v2_norm
        self.PREPROCESS_FEATURES = [
            "fast_signal_20260310_v1_norm", "buy_1m_ret_norm", "sell_1m_ret_norm",
            "pv_corr_40x3s", "KER_40x3s", "Vol_Squeeze", "vol_shock", "vol_feat_300s",
            "smart_momentum_300s", "Gini_300s", "pressure_60s", "cmf_v2_norm",
        ]
        self.STATIC_STATE_FEATURES = [
            # 静态特征 / 信号
            "fast_signal_20260310_v1_norm", "signal_1m_norm", 
            "pv_corr_40x3s", "KER_40x3s", "Vol_Squeeze", "vol_shock", "vol_feat_300s",
            "smart_momentum_300s", "Gini_300s", "pressure_60s", "cmf_v2_norm",
        ]
        self.DYNAMIC_STATE_FEATURES = [
            "pos_ratio", "time_ratio", "gap", "level_1_baseprice_bp","shadow_vwap_dev_bp",
            "sum_impact_bp", "now_slippage_bp","exp_slippage_bp","L1_capacity",
            "log_position"
        ]
        self.STATE_FEATURES = self.STATIC_STATE_FEATURES + self.DYNAMIC_STATE_FEATURES
        
        # 3. 动作空间定义 (Actor 仅输出意图，由 Env 映射为具体比例和档位限制)
        self.action_mapping:Dict[int, Tuple[float, int]] = {
            0: (0.0, 0),    
            1: (0.02, 1),   
            2: (0.05, 1),   
            3: (0.10, 1),   
            4: (0.20, 1),   
            5: (0.50, 10),  
            6: (0.75, 10),  
            7: (1.00, 10),  
        }
        
        # 4. 初始化 Sample Pool
        self.train_pool: SamplePool = SamplePool(
            meta_dir=cfg.meta_dir,
            case_dir=cfg.case_dir,
            start_date=cfg.train_start,
            end_date=cfg.train_end,
            max_limit=int(1e15)
        )
        self.test_pool: SamplePool = SamplePool(
            meta_dir=cfg.meta_dir,
            case_dir=cfg.case_dir,
            start_date=cfg.test_start,
            end_date=cfg.test_end,
            max_limit=int(1e15)
        )
        self.train_pool_size:int= self.train_pool.sample_pool_size
        self.test_pool_size:int = self.test_pool.sample_pool_size
        # 环境运行时变量
        self.history : List[np.ndarray] = []
        
    def reset(self, set_id: int = -1,mode: str = "train") -> Tuple[ARY, dict]:
        # 获取当前 Case 数据
        if mode not in ["train", "test"]:
            raise ValueError(f"Unknown mode: '{mode}'. Supported modes are 'train' and 'test'.")
        self.sample_pool: SamplePool = self.train_pool if mode == "train" else self.test_pool   
        self.case_info: Dict = self.sample_pool.get_sample(method=self.sample_method, set_id=set_id)
        
        # 加载 Tick 数据并截取当前 15min Window
        raw_df = pd.read_feather(self.case_info["file_path"])
        self.df: pd.DataFrame = raw_df  
        
        # 目标与进度初始化
        self.target_position: float = self.case_info["target_change"]
        assert self.target_position != 0.0, f"Target position cannot be zero in sample {self.case_info['id']}"
        self.realized_position: float = 0.0
        self.side: int = np.sign(self.target_position)
        self.uncompleted_flag: bool = False
        
        # 金融状态初始化
        self.sum_impact: float = 0.0
        self.base_price: float = self.df.iloc[0]['mid']
        self.total_amount: float = abs(self.target_position * self.base_price)
        self.actual_cashflow: float = 0.0  # 仅用于统计 info
        self.ideal_cashflow: float = 0.0
        self.global_cum_volume: float = 0.0
        self.global_cum_amount: float = 0.0
        self.actual_amount: float = 0.0
        
        # 游标初始化：留出最后一个 tick 作为强制结算专用的观测点
        self.time_idx: int = 0
        self.max_steps: int = len(self.df) - 1 #! 现在没有到下一个开始，检查下对不对
        
        self.history.clear()
        initial_state = self._get_state()
        
        while len(self.history) < self.K:
            if self.padding0:
                self.history.insert(0, np.zeros_like(initial_state))
            else:
                self.history.insert(0, initial_state)
        assert len(self.history) == self.K
        state_stack = np.concatenate(self.history, axis=0)
        return state_stack, self.case_info

    def _compute_l2_execution(self, row, exp_vol: float, max_level: int) -> Tuple[float, float]:
        """按照多档行情的实际量价进行撮合，考虑吃穿档"""
        if exp_vol == 0:
            return 0.0, 0.0
        is_buy = exp_vol > 0
        rem_vol = abs(exp_vol)
        total_cost = 0.0
        actual_vol = 0.0
        
        for i in range(1, max_level + 1):
            px = getattr(row, f"sale{i}" if is_buy else f"buy{i}", np.nan)
            v = getattr(row, f"sc{i}" if is_buy else f"bc{i}", np.nan)
            
            if pd.isna(px) or px == 0.0 or pd.isna(v) or v == 0.0: 
                continue
                
            if rem_vol <= v:
                total_cost += rem_vol * px
                actual_vol += rem_vol
                rem_vol = 0.0
                break
            else:
                total_cost += v * px
                actual_vol += v
                rem_vol -= v

        return total_cost, actual_vol * np.sign(exp_vol)

    def _compute_forced_execution_cost(self, row, exp_vol: float) -> float:
        """用于状态预期计算和结尾强平：扫过 10 档，若仍未吃完，加罚 0.01 强行吃掉"""
        if exp_vol == 0:
            return 0.0
        is_buy = exp_vol > 0
        rem_vol = abs(exp_vol)
        total_cost = 0.0
        
        last_valid_px = row['mid']  # 回退基准
        
        # 先扫 10 档
        for i in range(1, 11):
            px = getattr(row, f"sale{i}" if is_buy else f"buy{i}", np.nan)
            v = getattr(row, f"sc{i}" if is_buy else f"bc{i}", np.nan)
            
            if pd.isna(px) or px == 0.0 or pd.isna(v) or v == 0.0: 
                continue
                
            last_valid_px = px
            #print(f"Forced execution checking level {i}: price={px}, volume={v}, rem_vol={rem_vol}")
            if rem_vol <= v:
                total_cost += rem_vol * px
                rem_vol = 0.0
                break
            else:
                total_cost += v * px
                rem_vol -= v
            #print(f"After level {i}: total_cost={total_cost}, rem_vol={rem_vol}")
                
        # 10 档仍未吃完，第 11 档加罚吃穿所有剩余
        if rem_vol > 1e-8:
            penalty_px = last_valid_px + 0.01 if is_buy else last_valid_px - 0.01
            penalty_px = max(0.01, penalty_px)  # 防止极端情况价格穿零
            total_cost += rem_vol * penalty_px

        return total_cost
    def _trans_dict_norm(self, state_dict: Dict[str, float], side: int) -> Dict[str, float]:
        state_dict_norm = {}
        state_dict_norm["fast_signal_20260310_v1_norm"] = state_dict["fast_signal_20260310_v1_norm"] * side
        state_dict_norm["signal_1m_norm"] = (state_dict["buy_1m_ret_norm"] if side > 0 else state_dict["sell_1m_ret_norm"])
        state_dict_norm["pv_corr_40x3s"] = state_dict["pv_corr_40x3s"] * side
        state_dict_norm["KER_40x3s"] = state_dict["KER_40x3s"] * side
        state_dict_norm["Vol_Squeeze"] = state_dict["Vol_Squeeze"]
        state_dict_norm["vol_shock"] = state_dict["vol_shock"] 
        state_dict_norm["vol_feat_300s"] = state_dict["vol_feat_300s"]
        state_dict_norm["smart_momentum_300s"] = state_dict["smart_momentum_300s"] * side
        state_dict_norm["Gini_300s"] = state_dict["Gini_300s"]
        state_dict_norm["pressure_60s"] = state_dict["pressure_60s"] * side
        state_dict_norm["cmf_v2_norm"] = state_dict["cmf_v2_norm"]
        state_dict_norm["pos_ratio"] = state_dict["pos_ratio"]
        state_dict_norm["time_ratio"] = state_dict["time_ratio"]
        state_dict_norm["gap"] = state_dict["gap"]
        state_dict_norm["level_1_baseprice_bp"] = state_dict["level_1_baseprice_bp"] * side
        state_dict_norm["shadow_vwap_dev_bp"] = state_dict["shadow_vwap_dev_bp"] * side
        state_dict_norm["sum_impact_bp"] = state_dict["sum_impact_bp"] * side
        state_dict_norm["now_slippage_bp"] = state_dict["now_slippage_bp"]
        state_dict_norm["exp_slippage_bp"] = state_dict["exp_slippage_bp"]
        state_dict_norm["L1_capacity"] = state_dict["L1_capacity"]
        state_dict_norm["log_position"] = state_dict["log_position"]
        return state_dict_norm
    def _get_state(self) -> np.ndarray:
        row = self.df.iloc[self.time_idx]
        state_dict: Dict = {} 
        state_dict.update(row[self.PREPROCESS_FEATURES].to_dict())
       
        state_dict["pos_ratio"] = np.abs((self.realized_position) / (self.target_position))
        state_dict["time_ratio"] = self.time_idx / self.max_steps
        state_dict["gap"] = state_dict["pos_ratio"] - state_dict["time_ratio"]

        level_1_price = row['sale1'] if self.side > 0 else row['buy1']
        if pd.isna(level_1_price): level_1_price = row['mid']
        state_dict["level_1_baseprice_bp"] = ((level_1_price - self.base_price) / self.base_price * 10000)
        state_dict["shadow_vwap_dev_bp"] = (self.global_cum_amount / (self.global_cum_volume + 1e-8) - self.base_price) / self.base_price * 10000 if self.global_cum_volume > 1e-8 else 0.0
        state_dict["sum_impact_bp"] = self.sum_impact * 10000
        #print(self.actual_cashflow, self.ideal_cashflow, self.actual_amount)
        state_dict["now_slippage_bp"] = (self.actual_cashflow - self.ideal_cashflow) / (self.actual_amount + 1e-8) * 10000 if self.actual_amount > 1e-8 else 0.0
        # 3. Expected Slippage 预期滑点计算 (如果当前直接市价梭哈吃到底)
        uncompleted_pos = self.target_position - self.realized_position
        if abs(uncompleted_pos) > 1e-8:
            exp_cost = self._compute_forced_execution_cost(row, uncompleted_pos)
            # 使用现有冲击计算该强平产生的 cashflow
            exp_cashflow = -np.sign(uncompleted_pos) * exp_cost * (1 + self.sum_impact)
            ideal_exp_cashflow = -uncompleted_pos * self.base_price
            exp_pnl = exp_cashflow - ideal_exp_cashflow
            exp_slippage_bp = ((exp_pnl + (self.actual_cashflow - self.ideal_cashflow)) / (self.total_amount)) * 10000
        else:
            exp_slippage_bp = 0.0
        state_dict["exp_slippage_bp"] = exp_slippage_bp

        state_dict["L1_capacity"] = (row['sc1'] if self.side > 0 else row['bc1']) / (abs(self.target_position))
        state_dict["log_position"] = self.target_position

        #print(state_dict)
        state_dict_norm = self._trans_dict_norm(state_dict, self.side)
        
        state_arr = np.array([state_dict_norm[k] for k in self.STATE_FEATURES], dtype=np.float32)
        
        # Nan 处理
        if not np.isfinite(state_arr).all():
            raise ValueError(f"State contains non-finite values at time_idx {self.time_idx}: {state_dict}  --- {self.df.iloc[self.time_idx].to_dict()}")
            state_arr = np.nan_to_num(state_arr, nan=0.0, posinf=0.0, neginf=0.0)
            
        return state_arr

    def _check_trigger(self) -> bool:
        """
        Env 内部的 Actor 触发器，用于控制 Frame Skip (跳帧)
        返回 True 表示需要唤醒 Actor 进行动作决策
        """
        #row = self.df.iloc[self.time_idx]
        
        if abs(self.sum_impact) < 0.0002:
            return True
            
        """signal_1m = row['buy_1m_ret'] if self.side > 0 else row['sell_1m_ret']
        if signal_1m > 0.002:
            return True
        pos_ratio = self.realized_position / (self.target_position)
        time_ratio = self.time_idx / self.max_steps 
        if (time_ratio - pos_ratio) > 0.10:
            return True"""
            
        return False

    def step(self, action: int) -> Tuple[ARY, float, bool, bool, dict]:
        target_pct, max_level = self.action_mapping[int(action)]
        
        uncompleted_delta = self.target_position - self.realized_position
        exp_vol = self.target_position * target_pct
        # 截断限制
        if self.side > 0:
            exp_vol = min(exp_vol, uncompleted_delta)
        else:
            exp_vol = max(exp_vol, uncompleted_delta)
            
        # 1. 执行当前 Tick 的交易
        row = self.df.iloc[self.time_idx]
        base_cost, actual_vol = self._compute_l2_execution(row, exp_vol, max_level)
        #print("\n\n\nnow_step:", self.time_idx,row.time," action:", action, "exp_vol:", exp_vol, "actual_vol:", actual_vol, "base_cost:", base_cost)
        step_reward = 0.0
        if actual_vol != 0:
            # 简洁 PnL 计算：直接基于差额获得即时 Reward
            cashflow_change = -np.sign(actual_vol) * base_cost * (1 + self.sum_impact)
            ideal_cashflow_change = -actual_vol * self.base_price
             
            step_reward = cashflow_change - ideal_cashflow_change
            
            self.actual_cashflow += cashflow_change
            self.ideal_cashflow += ideal_cashflow_change
            self.realized_position += actual_vol
            self.actual_amount += abs(base_cost) * (1 + self.sum_impact) #! 在单次调仓的假设里 和 actual_cashflow 完全一致 只不过是abs累加
            cmf = row['cmf_v2']
            self.sum_impact += np.sign(actual_vol) * cmf * np.sqrt(abs(actual_vol))
            
        # 游标推进
        self.time_idx += 1
        self.sum_impact *= np.exp(-self.lambda_decay)
        self.global_cum_volume += self.df.iloc[self.time_idx].vol 
        self.global_cum_amount += self.df.iloc[self.time_idx].amount
        terminated = False
        truncated = False
        
        # 2. Frame Skip：如果直接到达了允许动作的最后一个有效步，强制唤醒
        if self.time_idx >= self.max_steps:
            terminated = True
        else:
            while not terminated and not self._check_trigger():
                self.time_idx += 1
                self.sum_impact *= np.exp(-self.lambda_decay)
                self.global_cum_volume += self.df.iloc[self.time_idx].vol
                self.global_cum_amount += self.df.iloc[self.time_idx].amount
                #print("Skipping to time_idx:", self.time_idx, self.df.iloc[self.time_idx].time, "sum_impact:", self.sum_impact)#!
                if self.time_idx >= self.max_steps:
                    terminated = True
                    break
            
        # 如果满仓也强制终止
        if abs(self.realized_position - self.target_position) < 1e-4:
            terminated = True
            
        # 3. 结算未完成部分 (十档 + 第十一档强平机制)
        if terminated:
            uncompleted_pos = self.target_position - self.realized_position
            if abs(uncompleted_pos) > 1e-4:
                self.uncompleted_flag = True
                
                # 取切片序列的绝对最后一个行作为强平参考物
                last_row = self.df.iloc[-1]
                
                # 计算强平总成本
                forced_cost = self._compute_forced_execution_cost(last_row, uncompleted_pos)
                
                penalty_cashflow = -np.sign(uncompleted_pos) * forced_cost * (1 + self.sum_impact)
                ideal_cost_for_uncompleted = -uncompleted_pos * self.base_price
                
                # 计算强平的即时惩罚，并累加到该 step_reward
                step_reward += (penalty_cashflow - ideal_cost_for_uncompleted)
                self.actual_cashflow += penalty_cashflow
                self.realized_position += uncompleted_pos
                self.ideal_cashflow += ideal_cost_for_uncompleted
                self.actual_amount += abs(forced_cost) * (1 + self.sum_impact)
                assert self.ideal_cashflow - (-self.target_position * self.base_price) < 1e-4, f"Ideal cashflow should be exactly target_position * base_price, but got {self.ideal_cashflow} vs {-self.target_position * self.base_price}"
            assert self.target_position - self.realized_position < 1e-4, f"At termination, uncompleted position should be close to zero, but got {self.target_position - self.realized_position}"
                
        # 整理输出状态
        new_state = self._get_state() if not terminated else self.history[-1] # 如果终止，状态变化已无意义，填补最后有效帧
        self.history.append(new_state)
        self.history.pop(0)
        state_stack = np.concatenate(self.history, axis=0)
        
        info = {
            "absolute_id": self.case_info["absolute_id"],
            "target_position": self.target_position,
        }
        
        return (state_stack, step_reward, terminated, truncated, info)

def check_stock_trading_env():
    import numpy as np
    import torch

    random.seed(527)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #actor_path = "/code/srwang/Finrl/log/20260219-131849_Agent_reinforce entropy0.0005  base_alpha0.1 lr1e-4 net-256-256-64 gpu0/actor__003683647488.pt"
    #device = f"cuda:0" if torch.cuda.is_available() else "cpu"
    #actor = torch.load(actor_path, map_location=device, weights_only=False)
    config_path = "/home/songrui.wang/elegantRL/envs/config_env.yaml"
    full_cfg = OmegaConf.load(config_path)
    env_cfg = full_cfg.env
    env = AStockExecutionEnv(env_cfg)
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
    )  # Example ID and index
    target_id = "100000792"
    set_id = -1
    for idx, sample in enumerate(env.sample_pool):
        if sample["id"] == target_id and sample["type"] == "short":
            set_id = idx
            break
    print("set_id:", set_id)"""
    case_idx = 0
    state, info = env.reset(set_id=case_idx, mode="train")
    print(info)
    # state, info = env.reset(sequential=True)  # Example ID and index
    # print("Initial State:", state)
    slippage_bp = []
    reward_cum = 0
    id = 0
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # for _ in tqdm(range(500000000), desc="Simulation Progress"):
    for _ in range(500000000):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        # print(_, "state=", state.cpu().numpy())
        # print(state.shape)
        # action = actor(state).detach().cpu().item()
        action = 1
        # print(action)
        # action = 0
        if _ > 2000000000:
            break
            # print("state: ", state)
            # print("Action:", rrr[action])
            ...
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

            if random.random() < 1:
                print(f"slippage={-reward_cum:.6f}, mean={np.mean(slippage_bp):.6f}")
            # print("-----------end-----------")
            id += 1
            case_idx += 1
            if id == 1000:
                break
            state, info = env.reset(set_id=case_idx, mode="train")
            #print(info)
            reward_cum = 0

    print("slippage value=", slippage_bp)
    print(np.mean(slippage_bp))
   # print(env.tot_uncompleted)


if __name__ == "__main__":
    check_stock_trading_env()