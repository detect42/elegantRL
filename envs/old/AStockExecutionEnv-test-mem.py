import os
import random
import time
import atexit
from typing import List, Tuple, Dict, Any
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch as th
import numba as nb
from omegaconf import DictConfig, OmegaConf
from multiprocessing import shared_memory, resource_tracker  # <-- 引入 resource_tracker
from tqdm import tqdm

from elegantRL.envs.sample_pool import SamplePool # 请确保这个路径正确

# ==============================================================================
# [新增] 全局预处理特征列表 (供 DataLoader 使用)
# ==============================================================================
PREPROCESS_FEATURES = [
    "fast_signal_20260310_v1_norm", "buy_1m_ret_norm", "sell_1m_ret_norm",
    "pv_corr_40x3s", "KER_40x3s", "Vol_Squeeze", "vol_shock", "vol_feat_300s",
    "smart_momentum_300s", "Gini_300s", "pressure_60s", "cmf_v2_norm",
]

# ==============================================================================
# [新增] 共享内存加载器 (DataLoader)
# ==============================================================================
def _load_single_file_to_shm(file_path: str):
    """独立函数：读取单文件并直接拼接写入 OS 共享内存"""
    try:
        df = pd.read_feather(file_path)
        if df.empty: return file_path, None

        arrays = {
            "mid": df["mid"].to_numpy(dtype=np.float64),
            "vwap": df["vwap"].to_numpy(dtype=np.float64),
            "vol": df["vol"].to_numpy(dtype=np.float64),
            "amount": df["amount"].to_numpy(dtype=np.float64),
            "cmf": df["cmf_v2"].to_numpy(dtype=np.float64),
            "ask_px": np.column_stack([df[f"sale{i}"].to_numpy(dtype=np.float64) for i in range(1, 11)]),
            "bid_px": np.column_stack([df[f"buy{i}"].to_numpy(dtype=np.float64) for i in range(1, 11)]),
            "ask_vol": np.column_stack([df[f"sc{i}"].to_numpy(dtype=np.int64) for i in range(1, 11)]),
            "bid_vol": np.column_stack([df[f"bc{i}"].to_numpy(dtype=np.int64) for i in range(1, 11)]),
            "features": np.column_stack([df[feat].to_numpy(dtype=np.float32) for feat in PREPROCESS_FEATURES])
        }

        meta = {}
        for key, arr in arrays.items():
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            
            # =====================================================================
            # 【核心修复】：告诉当前子进程的 tracker，解除对该共享内存的生命周期绑定！
            # 这样子进程死掉时，就不会把共享内存一起拖下水陪葬了。
            # =====================================================================
            resource_tracker.unregister(shm._name, 'shared_memory')
            
            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            shm_arr[:] = arr[:]  
            meta[key] = {
                "name": shm.name,
                "shape": arr.shape,
                "dtype": str(arr.dtype)
            }
            shm.close() 
        return file_path, meta
    except Exception as e:
        print(f"[Error] Load SHM Failed {file_path}: {e}")
        return file_path, None

def preload_data_to_shm(file_paths: List[str], max_workers=16) -> dict:
    """多进程并行将文件送入共享内存"""
    shm_metadata = {}
    print(f"\n🚀 正在并行拉取 {len(file_paths)} 个文件至 OS 共享内存 (/dev/shm) ...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_load_single_file_to_shm, file_paths), total=len(file_paths)))
    
    for path, meta in results:
        if meta is not None:
            shm_metadata[path] = meta
    print("✅ 共享内存装载完毕！\n")
    return shm_metadata

def cleanup_all_shm(shm_metadata: dict):
    """守护钩子：程序退出时清理内存"""
    print("\n🧹 正在打扫 OS 共享内存，请勿强行关闭终端...")
    for file_path, metas in shm_metadata.items():
        for key, meta in metas.items():
            try:
                shm = shared_memory.SharedMemory(name=meta["name"])
                shm.unlink()
            except Exception:
                pass
    print("✨ 内存清理完成！")

# 全局随机种子
np.random.seed(527)
random.seed(527)
ARY = np.ndarray

@nb.njit(cache=True)
def _numba_l2_execution(px_levels: np.ndarray, vol_levels: np.ndarray, exp_vol: float, max_level: int) -> Tuple[float, float]:
    if exp_vol == 0.0 or max_level <= 0:
        return 0.0, 0.0
    is_buy = exp_vol > 0.0
    sign = 1.0 if is_buy else -1.0
    rem_vol = abs(exp_vol)
    total_cost = 0.0
    actual_vol = 0.0
    upper = max_level if max_level < 10 else 10
    for i in range(upper):
        px = px_levels[i]
        v = vol_levels[i]
        if np.isnan(px) or px == 0.0 or np.isnan(v) or v == 0.0:
            continue
        if rem_vol <= v:
            total_cost += rem_vol * px
            actual_vol += rem_vol
            rem_vol = 0.0
            break
        total_cost += v * px
        actual_vol += v
        rem_vol -= v
    return total_cost, actual_vol * sign

@nb.njit(cache=True)
def _numba_forced_execution_cost(px_levels: np.ndarray, vol_levels: np.ndarray, mid_px: float, exp_vol: float) -> float:
    if exp_vol == 0.0:
        return 0.0
    is_buy = exp_vol > 0.0
    rem_vol = abs(exp_vol)
    total_cost = 0.0
    last_valid_px = mid_px
    for i in range(10):
        px = px_levels[i]
        v = vol_levels[i]
        if np.isnan(px) or px == 0.0 or np.isnan(v) or v == 0.0:
            continue
        last_valid_px = px
        if rem_vol <= v:
            total_cost += rem_vol * px
            rem_vol = 0.0
            break
        total_cost += v * px
        rem_vol -= v
    if rem_vol > 1e-8:
        penalty_px = (last_valid_px + 0.01) if is_buy else (last_valid_px - 0.01)
        penalty_px = max(0.01, penalty_px)
        total_cost += rem_vol * penalty_px
    return total_cost

@nb.njit(cache=True)
def _numba_fast_forward(time_idx: int,
                        max_steps: int,
                        sum_impact: float,
                        impact_decay: float,
                        vol_arr: np.ndarray,
                        amount_arr: np.ndarray) -> Tuple[int, float, float, float, bool]:
    add_vol = 0.0
    add_amt = 0.0
    terminated = False
    time_idx += 1
    sum_impact *= impact_decay
    add_vol += vol_arr[time_idx]
    add_amt += amount_arr[time_idx]
    if time_idx >= max_steps:
        terminated = True
        return time_idx, sum_impact, add_vol, add_amt, terminated
    while abs(sum_impact) >= 1000.0002: # 保持你的原有逻辑
        time_idx += 1
        sum_impact *= impact_decay
        add_vol += vol_arr[time_idx]
        add_amt += amount_arr[time_idx]
        if time_idx >= max_steps:
            terminated = True
            break
    return time_idx, sum_impact, add_vol, add_amt, terminated


class AStockExecutionEnv:
    def __init__(self, cfg: DictConfig):
        # 1. 配置参数提取
        self.num_envs = cfg.num_envs
        self.K: int = cfg.K
        self.state_dim: int = cfg.state_dim
        self.action_dim: int = cfg.action_dim
        self.lambda_decay: float = cfg.lambda_decay
        self.sample_method: str = cfg.sample_method
        self.if_discrete: bool = cfg.if_discrete
        self.padding0: bool = cfg.padding0
        self.max_step: int = cfg.max_step
        self.eval_num_workers: int = cfg.eval_num_workers
        self.noise_std_ratio: float = cfg.noise_std_ratio
        self.device = th.device("cpu") if cfg.gpu_id == -1 else th.device(f"cuda:{cfg.gpu_id}")
        self.mode: str = cfg.mode 
        self.env_name: str = "AStockExecutionEnv"
        
        self.PREPROCESS_FEATURES = PREPROCESS_FEATURES
        self.STATIC_STATE_FEATURES = [
            "fast_signal_20260310_v1_norm", "signal_1m_norm", 
            "pv_corr_40x3s", "KER_40x3s", "Vol_Squeeze", "vol_shock", "vol_feat_300s",
            "smart_momentum_300s", "Gini_300s", "pressure_60s", "cmf_v2_norm",
        ]
        self.DYNAMIC_STATE_FEATURES = [
            "pos_ratio", "time_ratio", "gap", "level_1_baseprice_bp","shadow_vwap_dev_bp",
            "sum_impact_bp", "now_slippage_bp","exp_slippage_bp","L1_capacity", "log_position"
        ]
        self.STATE_FEATURES = self.STATIC_STATE_FEATURES + self.DYNAMIC_STATE_FEATURES
        
        self.action_mapping:Dict[int, Tuple[float, int]] = {
            0: (0.0, 0), 1: (0.02, 1), 2: (0.05, 1), 3: (0.10, 1), 
            4: (0.20, 1), 5: (0.50, 10), 6: (0.75, 10), 7: (1.00, 10),
        }
        
        self.train_pool: SamplePool = SamplePool(
            meta_dir=cfg.meta_dir, case_dir=cfg.case_dir, start_date=cfg.train_start, end_date=cfg.train_end, max_limit=int(1e15)
        )
        self.test_pool: SamplePool = SamplePool(
            meta_dir=cfg.meta_dir, case_dir=cfg.case_dir, start_date=cfg.test_start, end_date=cfg.test_end, max_limit=int(1e15)
        )
        self.train_pool_size:int= self.train_pool.sample_pool_size
        self.test_pool_size:int = self.test_pool.sample_pool_size
        
        self.history : List[np.ndarray] = []
        self.last_read_file_time: float = 0.0
        self.case_timing: Dict[str, float] = {
            "compute_l2_execution_s": 0.0,
            "compute_forced_execution_cost_s": 0.0,
            "get_state_s": 0.0,
        }

        self._l2_levels: int = 10
        self._impact_decay: float = np.exp(-self.lambda_decay)
        self._state_index: Dict[str, int] = {k: i for i, k in enumerate(self.STATE_FEATURES)}

        # [新增] SHM 元数据与生命周期管理
        self.shm_metadata: dict = {}
        self._active_shms: list = []

    def _close_shm(self):
        """释放上一个 case 的内存指针，防内存泄漏"""
        for shm in self._active_shms:
            shm.close()
        self._active_shms.clear()

    def _map_shm(self, meta_info: dict) -> np.ndarray:
        """核心：通过取件码，0 毫秒瞬间贴上物理内存"""
        shm = shared_memory.SharedMemory(name=meta_info["name"])
        arr = np.ndarray(meta_info["shape"], dtype=meta_info["dtype"], buffer=shm.buf)
        arr.flags.writeable = False # 绝对防御：必须只读
        self._active_shms.append(shm)
        return arr

    def reset(self, set_id: int = -1,mode: str = "train") -> Tuple[ARY, dict]:
        if mode not in ["train", "test"]:
            raise ValueError(f"Unknown mode: '{mode}'. Supported modes are 'train' and 'test'.")
        
        # 1. 获取当前 Case 路径
        self.sample_pool: SamplePool = self.train_pool if mode == "train" else self.test_pool   
        self.case_info: Dict = self.sample_pool.get_sample(method=self.sample_method, set_id=set_id)
        file_path = self.case_info["file_path"]
        
        # 2. === 彻底消灭 pd.read_feather ===
        read_t0 = time.perf_counter()
        
        self._close_shm() # 关闭旧内存
        meta = self.shm_metadata[file_path] # 拿到取件码
        
        # 极速映射 (替代了原来的 _build_numpy_cache)
        self._mid_arr = self._map_shm(meta["mid"])
        self._vwap_arr = self._map_shm(meta["vwap"])
        self._vol_arr = self._map_shm(meta["vol"])
        self._amount_arr = self._map_shm(meta["amount"])
        self._cmf_arr = self._map_shm(meta["cmf"])
        self._ask_px_arr = self._map_shm(meta["ask_px"])
        self._bid_px_arr = self._map_shm(meta["bid_px"])
        self._ask_vol_arr = self._map_shm(meta["ask_vol"])
        self._bid_vol_arr = self._map_shm(meta["bid_vol"])
        self._feature_arr = self._map_shm(meta["features"])
        
        self.last_read_file_time = time.perf_counter() - read_t0
        
        self.case_timing["compute_l2_execution_s"] = 0.0
        self.case_timing["compute_forced_execution_cost_s"] = 0.0
        self.case_timing["get_state_s"] = 0.0
        
        # 3. 目标与进度初始化
        self.target_position: float = self.case_info["target_change"]
        assert self.target_position != 0.0, f"Target position cannot be zero in sample {self.case_info['id']}"
        self.realized_position: float = 0.0
        self.side: int = np.sign(self.target_position)
        self.uncompleted_flag: bool = False
        
        # 金融状态初始化
        self.sum_impact: float = 0.0
        self.base_price: float = self._mid_arr[0]      # 替代了 self.df.iloc[0]['mid']
        self.total_amount: float = abs(self.target_position * self.base_price)
        self.actual_cashflow: float = 0.0  
        self.ideal_cashflow: float = 0.0
        self.global_cum_volume: float = 0.0
        self.global_cum_amount: float = 0.0
        self.actual_amount: float = 0.0
        
        # 时间指针初始化
        self.time_idx: int = 0
        self.max_steps: int = len(self._mid_arr) - 1   # 替代了 len(self.df) - 1

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

    def _compute_l2_execution(self, exp_vol: float, max_level: int) -> Tuple[float, float]:
        fn_t0 = time.perf_counter()
        if exp_vol == 0:
            self.case_timing["compute_l2_execution_s"] += (time.perf_counter() - fn_t0)
            return 0.0, 0.0

        idx = self.time_idx
        if exp_vol > 0:
            px_levels = self._ask_px_arr[idx]
            vol_levels = self._ask_vol_arr[idx]
        else:
            px_levels = self._bid_px_arr[idx]
            vol_levels = self._bid_vol_arr[idx]
        total_cost, actual_vol = _numba_l2_execution(px_levels, vol_levels, float(exp_vol), int(max_level))
        self.case_timing["compute_l2_execution_s"] += (time.perf_counter() - fn_t0)
        return total_cost, actual_vol

    def _compute_forced_execution_cost(self, exp_vol: float) -> float:
        fn_t0 = time.perf_counter()
        if exp_vol == 0:
            self.case_timing["compute_forced_execution_cost_s"] += (time.perf_counter() - fn_t0)
            return 0.0

        idx = self.time_idx
        if exp_vol > 0:
            px_levels = self._ask_px_arr[idx]
            vol_levels = self._ask_vol_arr[idx]
        else:
            px_levels = self._bid_px_arr[idx]
            vol_levels = self._bid_vol_arr[idx]
        total_cost = _numba_forced_execution_cost(px_levels, vol_levels, self._mid_arr[idx], float(exp_vol))
        self.case_timing["compute_forced_execution_cost_s"] += (time.perf_counter() - fn_t0)
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
        fn_t0 = time.perf_counter()
        idx = self.time_idx
        feat = self._feature_arr[idx]

        state_dict: Dict = {}
        for i, k in enumerate(self.PREPROCESS_FEATURES):
            state_dict[k] = feat[i]

        state_dict["pos_ratio"] = np.abs((self.realized_position) / (self.target_position))
        state_dict["time_ratio"] = self.time_idx / self.max_steps
        state_dict["gap"] = state_dict["pos_ratio"] - state_dict["time_ratio"]

        level_1_price = self._ask_px_arr[idx, 0] if self.side > 0 else self._bid_px_arr[idx, 0]
        if np.isnan(level_1_price):
            level_1_price = self._mid_arr[idx]
        state_dict["level_1_baseprice_bp"] = ((level_1_price - self.base_price) / self.base_price * 10000)
        state_dict["shadow_vwap_dev_bp"] = (self.global_cum_amount / (self.global_cum_volume + 1e-8) - self.base_price) / self.base_price * 10000 if self.global_cum_volume > 1e-8 else 0.0
        state_dict["sum_impact_bp"] = self.sum_impact * 10000
        state_dict["now_slippage_bp"] = (self.actual_cashflow - self.ideal_cashflow) / (self.actual_amount + 1e-8) * 10000 if self.actual_amount > 1e-8 else 0.0
        
        uncompleted_pos = self.target_position - self.realized_position
        if abs(uncompleted_pos) > 1e-6:
            exp_cost = self._compute_forced_execution_cost(uncompleted_pos)
            exp_cashflow = -np.sign(uncompleted_pos) * exp_cost * (1 + self.sum_impact)
            ideal_exp_cashflow = -uncompleted_pos * self.base_price
            exp_pnl = exp_cashflow - ideal_exp_cashflow
            exp_slippage_bp = ((exp_pnl + (self.actual_cashflow - self.ideal_cashflow)) / (self.total_amount)) * 10000
        else:
            exp_slippage_bp = 0.0
        state_dict["exp_slippage_bp"] = exp_slippage_bp

        state_dict["L1_capacity"] = (self._ask_vol_arr[idx, 0] if self.side > 0 else self._bid_vol_arr[idx, 0]) / (abs(self.target_position))
        state_dict["log_position"] = self.target_position

        state_dict_norm = self._trans_dict_norm(state_dict, self.side)
        state_arr = np.array([state_dict_norm[k] for k in self.STATE_FEATURES], dtype=np.float32)
        
        if not np.isfinite(state_arr).all():
            state_arr = np.nan_to_num(state_arr, nan=0.0, posinf=0.0, neginf=0.0)

        self.case_timing["get_state_s"] += (time.perf_counter() - fn_t0)
        return state_arr

    def step(self, action: int) -> Tuple[ARY, float, bool, bool, dict]:
        action = 4 #! debug
        target_pct, max_level = self.action_mapping[int(action)]
        
        uncompleted_delta = self.target_position - self.realized_position
        exp_vol = self.target_position * target_pct

        if self.side > 0:
            exp_vol = min(exp_vol, uncompleted_delta)
        else:
            exp_vol = max(exp_vol, uncompleted_delta)
            
        base_cost, actual_vol = self._compute_l2_execution(exp_vol, max_level)
        step_reward = 0.0
        if actual_vol != 0:
            cashflow_change = -np.sign(actual_vol) * base_cost * (1 + self.sum_impact)
            ideal_cashflow_change = -actual_vol * self.base_price
             
            step_reward = cashflow_change - ideal_cashflow_change
            
            self.actual_cashflow += cashflow_change
            self.ideal_cashflow += ideal_cashflow_change
            self.realized_position += actual_vol
            self.actual_amount += abs(base_cost) * (1 + self.sum_impact) 
            cmf = self._cmf_arr[self.time_idx]
            self.sum_impact += np.sign(actual_vol) * cmf * np.sqrt(abs(actual_vol))

        self.time_idx, self.sum_impact, add_vol, add_amt, terminated = _numba_fast_forward(
            self.time_idx, self.max_steps, self.sum_impact, self._impact_decay, self._vol_arr, self._amount_arr,
        )
        self.global_cum_volume += add_vol
        self.global_cum_amount += add_amt
        truncated = False
            
        if abs(self.realized_position - self.target_position) < 1e-4:
            terminated = True
            
        # 结算未完成部分11档强平
        if terminated:
            uncompleted_pos = self.target_position - self.realized_position
            if abs(uncompleted_pos) > 1e-4:
                self.uncompleted_flag = True
                forced_cost = self._compute_forced_execution_cost(uncompleted_pos)
                penalty_cashflow = -np.sign(uncompleted_pos) * forced_cost * (1 + self.sum_impact)
                ideal_cost_for_uncompleted = -uncompleted_pos * self.base_price
                
                step_reward += (penalty_cashflow - ideal_cost_for_uncompleted)
                self.actual_cashflow += penalty_cashflow
                self.realized_position += uncompleted_pos
                self.ideal_cashflow += ideal_cost_for_uncompleted
                self.actual_amount += abs(forced_cost) * (1 + self.sum_impact)
                assert self.ideal_cashflow - (-self.target_position * self.base_price) < 1e-4, f"Ideal cashflow error"
            assert self.target_position - self.realized_position < 1e-4, f"Uncompleted pos error"
                
        # 整理输出状态
        new_state = self._get_state() if not terminated else self.history[-1] 
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
    config_path = "/home/songrui.wang/elegantRL/envs/config_env.yaml"
    full_cfg = OmegaConf.load(config_path)
    env_cfg = full_cfg.env
    env = AStockExecutionEnv(env_cfg)
    env.if_discrete = True
    
    # =========================================================
    # [新增] 截获当前测试脚本需要的所有 case，统一执行并行内存装载
    # =========================================================
    all_files = set()
    # 为了跑通测试脚本，我们把需要的 case 扫一遍
    if hasattr(env.train_pool, 'sample_pool'):
        for s in env.train_pool.sample_pool: all_files.add(s["file_path"])
    if hasattr(env.test_pool, 'sample_pool'):
        for s in env.test_pool.sample_pool: all_files.add(s["file_path"])
        
    all_files = list(all_files)[:] # 如果测试，可以仅限制预加载前两千个 case，防止撑爆节点内存
    
    # 启动预加载，并挂载元数据到环境
    shm_metadata = preload_data_to_shm(all_files, max_workers=16)
    env.shm_metadata = shm_metadata
    
    # 过滤 sample_pool 确保它只会抽到已经成功 load 进内存的 case
    if hasattr(env.train_pool, 'sample_pool'):
        env.train_pool.sample_pool = [s for s in env.train_pool.sample_pool if s["file_path"] in shm_metadata]
    
    # 注册清理钩子
    atexit.register(cleanup_all_shm, shm_metadata)
    # =========================================================

    def _safe_pct(x: float, y: float) -> float:
        return (x / y * 100.0) if y > 1e-12 else 0.0

    case_total_t0 = time.perf_counter()
    state, info = env.reset(mode="train")  
    
    slippage_bp = []
    reward_cum = 0
    id = 0
    device = "cpu"
    cum_read_s = 0.0
    cum_l2_forced_s = 0.0
    cum_get_state_s = 0.0
    cum_other_s = 0.0
    cum_total_s = 0.0
    for _ in range(500000000):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        action = 4 #! Debug action
        
        if _ > 2000000:
            break

        state, reward, terminated, truncated, info = env.step(action)
        reward_cum += reward
        
        if terminated or truncated:
            slippage_bp.append(-reward_cum)
            case_total_s = time.perf_counter() - case_total_t0
            case_l2_s = env.case_timing["compute_l2_execution_s"]
            case_forced_s = env.case_timing["compute_forced_execution_cost_s"]
            case_l2_forced_s = case_l2_s + case_forced_s
            case_get_state_s = env.case_timing["get_state_s"]
            case_read_s = env.last_read_file_time
            case_other_s = max(0.0, case_total_s - case_read_s - case_l2_forced_s - case_get_state_s)

            cum_read_s += case_read_s
            cum_l2_forced_s += case_l2_forced_s
            cum_get_state_s += case_get_state_s
            cum_other_s += case_other_s
            cum_total_s += case_total_s

            if random.random() < 1:
                print(f"now_step={_}")
                print(f"slippage={-reward_cum:.6f}, mean={np.mean(slippage_bp):.6f}")
                print(
                    f"[case {id + 1:04d}] read={case_read_s:.6f}s ({_safe_pct(case_read_s, case_total_s):.2f}%), "
                    f"l2_forced={case_l2_forced_s:.6f}s ({_safe_pct(case_l2_forced_s, case_total_s):.2f}%), "
                    f"get_state={case_get_state_s:.6f}s ({_safe_pct(case_get_state_s, case_total_s):.2f}%), "
                    f"other={case_other_s:.6f}s ({_safe_pct(case_other_s, case_total_s):.2f}%), "
                    f"total={case_total_s:.6f}s"
                )
                print(
                    f"[cum  {id + 1:04d}] read={cum_read_s:.6f}s ({_safe_pct(cum_read_s, cum_total_s):.2f}%), "
                    f"l2_forced={cum_l2_forced_s:.6f}s ({_safe_pct(cum_l2_forced_s, cum_total_s):.2f}%), "
                    f"get_state={cum_get_state_s:.6f}s ({_safe_pct(cum_get_state_s, cum_total_s):.2f}%), "
                    f"other={cum_other_s:.6f}s ({_safe_pct(cum_other_s, cum_total_s):.2f}%), "
                    f"total={cum_total_s:.6f}s\n"
                )
            id += 1
            if id == 1000:
                break
            case_total_t0 = time.perf_counter()
            state, info = env.reset(mode="train")
            reward_cum = 0

    print(np.mean(slippage_bp))

if __name__ == "__main__":
    check_stock_trading_env()