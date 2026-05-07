# elegantRL/envs/shm_utils.py
import os
import pickle
import pandas as pd
import numpy as np
from typing import List
from multiprocessing import shared_memory, resource_tracker
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import signal
# 这里必须和 AStockExecutionEnv 中的保持绝对一致
PREPROCESS_FEATURES = [
    "fast_signal_20260310_v1_norm", "buy_1m_ret_norm", "sell_1m_ret_norm",
    "vwap_6s_over_60s", "vwap_rank_60s", "vwap_30s_over_300s", "vwap_rank_360s",
    "pv_corr_40x3s", "KER_40x3s", "Vol_Squeeze", "vol_shock", "vol_feat_300s",
    "smart_momentum_300s", "Gini_300s", "pressure_60s", "cmf_v2_norm",
    "micro_rv_20t_bp", "local_amplitude_20t_bp", "micro_price_location_20t", "vol_pulse_20t", "spread_bp", "OIR_raw", "ask_L2_defense", "bid_L2_defense"
]

def _load_single_file_to_shm(file_path: str):
    """读取单文件并写入共享内存 (带有 resource_tracker 解绑)"""
    try:
        df = pd.read_feather(file_path)
        if df.empty: return file_path, None

        arrays = {
            "real_time": df["time"].to_numpy(dtype=np.int64),
            "mid": df["mid"].to_numpy(dtype=np.float64),
            "vwap": df["vwap"].to_numpy(dtype=np.float64),
            "vol": df["vol"].to_numpy(dtype=np.float64),
            "amount": df["amount"].to_numpy(dtype=np.float64),
            "cmf": df["cmf_v2"].to_numpy(dtype=np.float64),
            "ask_px": np.column_stack([df[f"sale{i}"].to_numpy(dtype=np.float64) for i in range(1, 11)]),
            "bid_px": np.column_stack([df[f"buy{i}"].to_numpy(dtype=np.float64) for i in range(1, 11)]),
            "ask_vol": np.column_stack([df[f"sc{i}"].to_numpy(dtype=np.int64) for i in range(1, 11)]),
            "bid_vol": np.column_stack([df[f"bc{i}"].to_numpy(dtype=np.int64) for i in range(1, 11)]),
            "buy_1m_ret": df["buy_1m_ret"].to_numpy(dtype=np.float32),
            "sell_1m_ret": df["sell_1m_ret"].to_numpy(dtype=np.float32),
            "buy_1m_ret_norm": df["buy_1m_ret_norm"].to_numpy(dtype=np.float32),
            "sell_1m_ret_norm": df["sell_1m_ret_norm"].to_numpy(dtype=np.float32),
            "30m_ret_raw": df["fast_signal_20260310_v1"].to_numpy(dtype=np.float32),
            "features": np.column_stack([df[feat].to_numpy(dtype=np.float32) for feat in PREPROCESS_FEATURES])
        }

        meta = {}
        for key, arr in arrays.items():
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
            # 【关键】解绑子进程的生命周期
            resource_tracker.unregister(shm._name, 'shared_memory')
            
            shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
            shm_arr[:] = arr[:]  
            meta[key] = {"name": shm.name, "shape": arr.shape, "dtype": str(arr.dtype)}
            shm.close() 
        return file_path, meta
    except Exception as e:
        print(f"[Error] Load SHM Failed {file_path}: {e}")
        return file_path, None

def preload_data_to_shm(file_paths: List[str], max_workers=64) -> dict:
    """
    修改版：并行加载数据，并最终将元数据字典也存入共享内存索引。
    返回：包含索引块信息的字典 (很小)
    """
    raw_metadata = {}
    print(f"\n🚀 正在并行拉取 {len(file_paths)} 个文件至共享内存...")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(_load_single_file_to_shm, file_paths), total=len(file_paths)))
    
    for path, meta in results:
        if meta is not None:
            raw_metadata[path] = meta
            
    # --- 新增：将 raw_metadata 索引本身也塞进共享内存 ---
    meta_pickle = pickle.dumps(raw_metadata)
    # 创建一个唯一的索引块名字
    index_shm_name = f"shm_index_{os.getpid()}"
    index_shm = shared_memory.SharedMemory(create=True, size=len(meta_pickle), name=index_shm_name)
    # 解绑子进程追踪
    resource_tracker.unregister(index_shm._name, 'shared_memory')
    
    # 写入二进制数据
    index_shm.buf[:len(meta_pickle)] = meta_pickle
    index_shm.close()
    
    print(f"✅ 数据加载完成。索引大小: {len(meta_pickle)/1024/1024:.2f} MB")
    
    # 返回给 main 的是一个微小的凭证，记录了索引块在哪里，以及里面有多少个文件
    return {
        "index_shm_name": index_shm_name,
        "index_shm_size": len(meta_pickle),
        "file_count": len(raw_metadata),
        "raw_metadata": raw_metadata # 留给 cleanup_all_shm 使用
    }

def cleanup_all_shm(shm_info: dict):
    """加固后的清理函数，支持清理索引块和所有文件块"""
    if not shm_info: return
    raw_metadata = shm_info.get("raw_metadata", {})
    index_name = shm_info.get("index_shm_name")
    
    print(f"\n🧹 收到清理信号，准备释放内存...")
    
    # 1. 清理索引块
    if index_name:
        try:
            s = shared_memory.SharedMemory(name=index_name)
            s.unlink()
        except: pass

    # 2. 批量清理文件块
    for file_path, metas in raw_metadata.items():
        for key, meta in metas.items():
            try:
                s = shared_memory.SharedMemory(name=meta["name"])
                s.unlink()
            except: pass
    print("✨ 共享内存已彻底释放。")