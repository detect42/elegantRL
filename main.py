import json
import os
import shutil

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# 引入训练入口
from elegantrl import train_agent

# 引入我们重构好的逻辑工厂 (位于 elegantrl/train/config.py)
from elegantrl.train.config import init_before_training, process_config

# 假设这些文件还在原位，用于备份逻辑
# 注意：不需要再 import FutureExecEnv 类了，因为 process_config 会动态加载它


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    # ============================================================
    # 1. 逻辑加工工厂 (Logic Factory)
    # ============================================================
    # 这一步会自动：
    # - 计算 state_dim = 15 * K
    # - 推断 if_off_policy
    # - 加载 Agent 和 Env 的类对象
    # - 生成 cfg.env_args 字典
    cfg = process_config(cfg)
    OmegaConf.resolve(cfg)
    # ============================================================
    # 2. 环境初始化 (Side Effects)
    # ============================================================
    # 设置随机种子、PyTorch 线程、记录 CWD
    init_before_training(cfg)

    # ============================================================
    # 3. 兼容性操作：保存 config.json
    # ============================================================
    # 虽然 Hydra 已经保存了 config.yaml，但为了兼容你的习惯，我们存一份 json
    # OmegaConf.to_container 将 DictConfig 转为纯 Python 字典
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    # 打印一下看看 (可选)
    # print("Config dictionary:", json.dumps(config_dict, indent=2, default=str))

    config_json_path = os.path.join(cfg.eval.cwd, "config.json")
    with open(config_json_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    print(f"| Config saved to {os.path.join(os.getcwd(), 'config.json')}")

    # ============================================================
    # 4. 源码备份 (Source Code Backup)
    # ============================================================
    backup_source_code(cfg=cfg)

    # ============================================================
    # 5. 启动训练
    # ============================================================
    print(f"| Training Start: {cfg.agent.agent_name} in {cfg.eval.cwd}")

    # 这里的 cfg 已经是分层结构 (Hierarchical)，且包含所有必要参数
    # 请确保你已经修改了 train_agent 内部代码以支持 cfg.train.batch_size 这种访问方式
    train_agent(args=cfg)


def backup_source_code(cfg: DictConfig):
    """
    备份源代码到当前 Log 目录。
    注意：由于 Hydra 切换了工作目录，必须使用 get_original_cwd() 找到源码位置。
    """
    orig_cwd = "/code/srwang/Finrl"

    # 定义需要备份的文件 (相对于项目根目录 /code/srwang/finrl/)
    # 如果 elegantrl 在上一级，需要用 ../
    files_to_backup = [
        # 入口文件 (假设当前文件名是 main.py 或 future_modsac_discrete.py)
        "main.py",
        # 环境文件
        "../elegantrl/envs/FutureExecEnv.py",
        # Agent 文件 (注意路径，如果 elegantrl 不在当前目录，要写对相对路径)
        # 假设 elegantrl 在 /code/srwang/elegantrl，而我们在 /code/srwang/finrl
        "../elegantrl/agents/AgentModSACDiscrete.py",
        "../elegantrl/agents/AgentPPO.py",
        "../elegantrl/agents/AgentBase.py",
        "../elegantrl/agents/AgentReinforce.py",
    ]

    print("| Backup files:")
    for file_rel_path in files_to_backup:
        # 拼接原始绝对路径
        src = os.path.normpath(os.path.join(orig_cwd, file_rel_path))

        if os.path.exists(src):
            # 目标文件名 (只取文件名，不带路径)
            dst_name = os.path.basename(src)
            # 特殊处理：你之前代码里把 FutureExecEnv_v8.py 改名为了 FutureExecEnv.py
            if "FutureExecEnv" in dst_name:
                dst_name = "FutureExecEnv.py"
            dst_base = HydraConfig.get().runtime.output_dir
            dst = os.path.join(dst_base, dst_name)
            shutil.copy(src, dst)
            print(f"  -> {dst_name}")
        else:
            print(f"  [Warning] Source file not found: {src}")


if __name__ == "__main__":
    main()
