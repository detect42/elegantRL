import os
import time
from typing import List, Tuple, Dict
from typing import Callable, Union, Any
import time
import numpy as np
import torch as th
from omegaconf import DictConfig, OmegaConf
from threadpoolctl import threadpool_limits

TEN = th.Tensor


def _eval_worker(actor_cpu, env, case_ids, max_step, device_type="cpu", mode: str = "eval"):
    """
    运行在子进程中的评测逻辑
    """
    th.set_num_threads(1)

    with threadpool_limits(limits=4, user_api="blas"):
        # 确保 actor 在 CPU 上，并且是评估模式
        actor_cpu.eval()

        results = []
        action_counts = None
        if env.if_discrete:
            action_dim = int(env.action_dim)
            if action_dim > 0:
                action_counts = np.zeros(action_dim, dtype=np.int64)

        # 为了避免每次 step 都创建 tensor 的开销，预先定义好 device
        device = th.device(device_type)

        with th.no_grad():
            for set_id in case_ids:
                # print(set_id)
                # === 核心修改：传入 seq 参数 ===
                # 假设您的 env.reset 支持 seq 参数
                state, _ = env.reset(set_id=set_id, mode=mode)
                cumulative_returns = 0.0
                episode_steps = 0

                for _ in range(max_step):
                    # 转 Tensor
                    tensor_state = th.as_tensor(state, dtype=th.float32, device=device).unsqueeze(0)

                    # 推理 (Batch=1, CPU)
                    tensor_action = actor_cpu(tensor_state)
                    action = tensor_action.detach().numpy()[0]  # 已经是 CPU 了

                    if action_counts is not None:
                        action_arr = np.asarray(action)
                        action_idx = int(action_arr.item())
                        if 0 <= action_idx < action_counts.shape[0]:
                            action_counts[action_idx] += 1

                    # 环境交互
                    #print(action)
                    state, reward, terminated, truncated, _ = env.step(action)
                    cumulative_returns += reward
                    episode_steps += 1

                    if terminated or truncated:
                        break

                # 记录结果 (reward, step)
                results.append((cumulative_returns, episode_steps))

    return results, action_counts


class Evaluator:
    def __init__(self, cwd: str, env, args: DictConfig, if_tensorboard: bool = False):
        self.cwd = cwd  # current working directory to save model
        self.env = env  # the env for Evaluator, `eval_env = env` in default
        self.agent_id = args.sys.gpu_id
        self.total_step = 0  # the total training step
        self.start_time = time.time()  # `used_time = time.time() - self.start_time`
        self.eval_times = args.eval.times  # number of times that get episodic cumulative return
        self.eval_per_step = args.eval.per_step  # evaluate the agent per training steps
        self.eval_step_counter = -self.eval_per_step  # `self.total_step > self.eval_step_counter + self.eval_per_step`

        self.save_gap = args.eval.save_gap
        self.save_counter = 0
        self.if_keep_save = args.eval.if_keep_save
        self.if_over_write = args.eval.if_over_write

        self.recorder_path = f"{cwd}/recorder.npy"
        self.recorder: List = []  # total_step, r_avg, r_std, critic_value, ...
        self.recorder_step = args.eval.record_step  # start recording after the exploration reaches this step.
        self.max_r = -np.inf
        print(
            "| Evaluator:"
            "\n| `step`: Number of samples, or total training steps, or running times of `env.step()`."
            "\n| `time`: Time spent from the start of training to this moment."
            "\n| `avgR`: Average value of cumulative rewards, which is the sum of rewards in an episode."
            "\n| `stdR`: Standard dev of cumulative rewards, which is the sum of rewards in an episode."
            "\n| `avgS`: Average of steps in an episode."
            "\n| `objC`: Objective of Critic network. Or call it loss function of critic network."
            "\n| `objA`: Objective of Actor network. It is the average Q value of the critic network."
            f"\n{'#' * 80}\n"
            f"{'ID':<3}{'Step':>8}{'Time':>8} |"
            f"{'avgR':>8}{'stdR':>7}{'avgS':>7}{'stdS':>6} |"
            f"{'expR':>8}{'objC':>7}{'objA':>7}{'etc.':>7}",
            flush=True,
        )
        assert type(env.num_envs) == int and env.num_envs >= 1
        # Allow get_cumulative_rewards_and_step to return either Tensor or Tuple[Tensor, Tensor]

        self.get_cumulative_rewards_and_step: Callable[[Any], Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]]
        self.tensorboard: Union[Any, None] = None

        if args.eval.eval_dataset_test_all:
            self.get_cumulative_rewards_and_step = self.get_cumulative_rewards_and_step_single_env_parallel
        elif env.num_envs == 1:  # get attribute
            self.get_cumulative_rewards_and_step = self.get_cumulative_rewards_and_step_single_env
        else:  # vectorized environment
            print("Evaluator: Vectorized Env Num =", env.num_envs, flush=True)
            self.get_cumulative_rewards_and_step = self.get_cumulative_rewards_and_step_vectorized_env

        if if_tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self.tensorboard = SummaryWriter(f"{cwd}/tensorboard")
        else:
            self.tensorboard = None

    def evaluate_and_save(self, actor: th.nn.Module, steps: int, exp_r: float, logging_dict: Dict[str, float]):
        # print("now_steps=",self.total_step,"eval_step_counter= ",self.eval_step_counter," target_steps:",self.eval_step_counter + self.eval_per_step, " add steps->",steps," exp_r->",exp_r,flush=True)

        self.total_step += steps  # update total training steps

        if self.total_step < self.recorder_step:
            return
        if self.total_step < self.eval_step_counter + self.eval_per_step:
            return
        self.eval_step_counter = self.total_step
        rewards_step_result = self.get_cumulative_rewards_and_step(actor)
        eval_action_pct = None
        test_action_pct = None
        if isinstance(rewards_step_result, (tuple, list)) and len(rewards_step_result) == 4:
            rewards_step_ten_eval, rewards_step_ten_test, eval_action_pct, test_action_pct = rewards_step_result
        elif isinstance(rewards_step_result, (tuple, list)) and len(rewards_step_result) == 3:
            rewards_step_ten_eval, rewards_step_ten_test, eval_action_pct = rewards_step_result
        elif isinstance(rewards_step_result, (tuple, list)) and len(rewards_step_result) == 2:
            rewards_step_ten_eval, rewards_step_ten_test = rewards_step_result
        else:
            rewards_step_ten_eval = rewards_step_result
            rewards_step_ten_test = None

        #print("eval_pool:", rewards_step_ten_eval.shape, flush=True)  # p
        #if rewards_step_ten_test is not None:
        #    print("test_pool:", rewards_step_ten_test.shape, flush=True)  # print
        eval_returns = rewards_step_ten_eval[:, 0]  # episodic cumulative returns of an
        eval_steps = rewards_step_ten_eval[:, 1]  # episodic step number
        avg_r = eval_returns.mean().item()
        std_r = eval_returns.std().item()
        avg_s = eval_steps.mean().item()
        std_s = eval_steps.std().item()

        avg_r_test: Union[float, None] = None
        std_r_test: Union[float, None] = None
        avg_s_test: Union[float, None] = None
        std_s_test: Union[float, None] = None
        if rewards_step_ten_test is not None:
            test_returns = rewards_step_ten_test[:, 0]
            test_steps = rewards_step_ten_test[:, 1]
            avg_r_test = test_returns.mean().item()
            std_r_test = test_returns.std().item()
            avg_s_test = test_steps.mean().item()
            std_s_test = test_steps.std().item()

        train_time = int(time.time() - self.start_time)
        obj_critic_avg = logging_dict["obj_critic_avg"]
        obj_actor_avg = logging_dict["obj_actor_avg"]
        value_tuple = [v for v in logging_dict.values() if isinstance(v, (int, float))]
        logging_str = logging_dict.get("action_show_str", "")

        """record the training information"""
        self.recorder.append((self.total_step, avg_r, std_r, exp_r, *value_tuple))  # update recorder
        if self.tensorboard:
            self.tensorboard.add_scalar("info/critic_loss_sample", obj_critic_avg, self.total_step)
            self.tensorboard.add_scalar("info/actor_obj_sample", -1 * obj_actor_avg, self.total_step)
            if "obj_entropy_avg" in logging_dict:
                self.tensorboard.add_scalar("info/entropy_avg_sample", logging_dict["obj_entropy_avg"], self.total_step)
            if "current_entropy" in logging_dict:
                self.tensorboard.add_scalar("info/entropy_avg_sample", logging_dict["current_entropy"], self.total_step)
            self.tensorboard.add_scalar("info/exp_reward_sample", exp_r, self.total_step)
            # self.tensorboard.add_scalar("info/critic_loss_time", obj_critic_avg, train_time)
            # self.tensorboard.add_scalar("info/actor_obj_time", -1 * obj_actor_avg, train_time)

            # 分级保存 eval / test
            self.tensorboard.add_scalar("eval_set/avg_reward_sample", avg_r, self.total_step)
            self.tensorboard.add_scalar("eval_set/std_reward_sample", std_r, self.total_step)
            self.tensorboard.add_scalar("eval_set/avg_step", avg_s, self.total_step)
            self.tensorboard.add_scalar("eval_set/std_step", std_s, self.total_step)
            # self.tensorboard.add_scalar("reward/eval/avg_reward_time", avg_r, train_time)
            # self.tensorboard.add_scalar("reward/eval/std_reward_time", std_r, train_time)
            # self.tensorboard.add_scalar("reward/eval/exp_reward_time", exp_r, train_time)

            if avg_r_test is not None:
                self.tensorboard.add_scalar("test_set/avg_reward_sample", avg_r_test, self.total_step)
                self.tensorboard.add_scalar("test_set/std_reward_sample", std_r_test, self.total_step)
                self.tensorboard.add_scalar("test_set/avg_step", avg_s_test, self.total_step)
                self.tensorboard.add_scalar("test_set/std_step", std_s_test, self.total_step)
                # self.tensorboard.add_scalar("reward/test/avg_reward_time", avg_r_test, train_time)
                # self.tensorboard.add_scalar("reward/test/std_reward_time", std_r_test, train_time)
                # self.tensorboard.add_scalar("reward/test/exp_reward_time", exp_r_test, train_time)

        """print some information to Terminal"""
        prev_max_r = self.max_r
        self.max_r = max(self.max_r, avg_r)  # update max average cumulative rewards
        eval_action_pct_str = ""
        test_action_pct_str = ""
        if eval_action_pct is not None:
            eval_action_pct_str = " eval_action_pct=[" + ", ".join(f"{p:.2f}%" for p in eval_action_pct) + "]"
        if test_action_pct is not None:
            test_action_pct_str += " test_action_pct=[" + ", ".join(f"{p:.2f}%" for p in test_action_pct) + "]"

        print(
            f"{self.agent_id:<3}{self.total_step:8.2e}{train_time:8.0f} |"
            f"{avg_r:8.2f}{std_r:7.1f}{avg_s:7.0f}{std_s:6.0f} |"
            f"{exp_r:8.2f}{''.join(f'{n:7.2f}' for n in value_tuple)} {logging_str}|",
            f"{avg_r_test:8.2f}{std_r_test:7.1f}{avg_s_test:7.0f}{std_s_test:6.0f}" if avg_r_test is not None else "",
            "\n",
            eval_action_pct_str + test_action_pct_str,
            flush=True,
        )

        if_save = avg_r > prev_max_r
        if if_save:
            self.save_training_curve_jpg()
        if not self.if_keep_save:
            return

        self.save_counter += 1
        actor_path = None
        if if_save:  # save checkpoint with the highest episode return
            if self.if_over_write:
                actor_path = f"{self.cwd}/actor.pt"
            else:
                actor_path = f"{self.cwd}/actor__{self.total_step:012}_{self.max_r:09.3f}.pt"

        elif self.save_counter >= self.save_gap:
            self.save_counter = 0
            if self.if_over_write:
                actor_path = f"{self.cwd}/actor.pt"
            else:
                actor_path = f"{self.cwd}/actor__{self.total_step:012}.pt"

        if actor_path:
            th.save(actor, actor_path)  # save policy network in *.pt
            self.save_training_curve_jpg()

    def save_or_load_recoder(self, if_save: bool):
        if if_save:
            recorder_ary = np.array(self.recorder)
            np.save(self.recorder_path, recorder_ary)
        elif os.path.exists(self.recorder_path):
            recorder = np.load(self.recorder_path)
            self.recorder = [tuple(i) for i in recorder]  # convert numpy to list
            self.total_step = self.recorder[-1][0]

    def get_cumulative_rewards_and_step_single_env(self, actor) -> TEN:
        rewards_steps_list = [Get_rewards_and_steps(self.env, actor) for _ in range(self.eval_times)]
        rewards_steps_eval = th.tensor(rewards_steps_list, dtype=th.float32)
        return rewards_steps_eval  # rewards_steps_eval.shape[1] == 2

    def get_cumulative_rewards_and_step_vectorized_env(self, actor) -> TEN:
        rewards_step_list = [
            Get_cumulative_rewards_and_step_from_vec_env(self.env, actor)
            for _ in range(max(1, self.eval_times // self.env.num_envs))
        ]
        rewards_step_list = sum(rewards_step_list, [])
        rewards_step_eval = th.tensor(rewards_step_list)
        return rewards_step_eval  # rewards_steps_ten.shape[1] == 2

    def get_cumulative_rewards_and_step_single_env_parallel(
        self, actor
    ) -> Tuple[TEN, TEN, Union[List[float], None], Union[List[float], None]]:
        rewards_step_eval_test = Get_cumulative_rewards_and_step_single_env_parallel(self.env, actor)
        return rewards_step_eval_test  # rewards_steps_ten.shape[1] == 2

    def save_training_curve_jpg(self):
        recorder = np.array(self.recorder)

        train_time = int(time.time() - self.start_time)
        total_step = int(self.recorder[-1][0])
        fig_title = f"step_time_maxR_{int(total_step)}_{int(train_time)}_{self.max_r:.3f}"

        draw_learning_curve(recorder=recorder, fig_title=fig_title, save_path=f"{self.cwd}/LearningCurve.jpg")
        np.save(self.recorder_path, recorder)  # save self.recorder for `draw_learning_curve()`


"""util"""


def Get_rewards_and_steps(env, actor, if_render: bool = False) -> Tuple[float, int]:
    """Usage
    eval_times = 4
    net_dim = 2 ** 7
    actor_path = './LunarLanderContinuous-v2_PPO_1/actor.pt'

    env = build_env(env_class=env_class, env_args=env_args)
    actor = agent(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id).act
    actor.load_state_dict(th.load(actor_path, map_location=lambda storage, loc: storage))

    r_s_ary = [get_episode_return_and_step(env, act) for _ in range(eval_times)]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    """
    max_step = env.max_step
    device = next(actor.parameters()).device  # net.parameters() is a Python generator.

    state, info_dict = env.reset()
    episode_steps = 0
    cumulative_returns = 0.0  # sum of rewards in an episode
    for episode_steps in range(max_step):
        tensor_state = th.as_tensor(state, dtype=th.float32, device=device).unsqueeze(0)
        tensor_action = actor(tensor_state)
        action = tensor_action.detach().cpu().numpy()[0]  # not need detach(), because using th.no_grad() outside
        # print("eval:",action)
        state, reward, terminated, truncated, _ = env.step(action)
        cumulative_returns += reward

        if if_render:
            env.render()
        if terminated or truncated:
            break
    else:
        print("| get_rewards_and_step: WARNING. max_step > 12345", flush=True)

    env_unwrapped = getattr(env, "unwrapped", env)
    cumulative_returns = getattr(env_unwrapped, "cumulative_returns", cumulative_returns)
    return cumulative_returns, episode_steps + 1


def Get_cumulative_rewards_and_step_from_vec_env(env, actor) -> List[Tuple[float, int]]:
    device = env.device
    env_num = env.num_envs
    max_step = env.max_step
    """get returns and dones (GPU)"""
    returns = th.empty((max_step, env_num), dtype=th.float32, device=device)
    dones = th.empty((max_step, env_num), dtype=th.bool, device=device)

    state, info_dict = env.reset()  # must reset in vectorized env
    # print(state)
    for t in range(max_step):
        action = actor(state.to(device))
        assert action.shape == (env_num,) if env.if_discrete else (env_num, env.action_dim)
        state, reward, terminal, truncate, info_dict = env.step(action)
        returns[t] = reward
        dones[t] = th.logical_or(terminal, truncate)

    """get cumulative returns and step"""
    if hasattr(env, "cumulative_returns"):  # GPU
        returns_step_list = [(ret, env.max_step) for ret in env.cumulative_returns]
    else:  # CPU
        returns = returns.cpu()
        dones = dones.cpu()

        returns_step_list = []
        for i in range(env_num):
            dones_where = th.where(dones[:, i].eq(1))[0] + 1
            episode_num = len(dones_where)
            if episode_num == 0:
                continue

            j0 = 0
            for j1 in dones_where.tolist():
                reward_sum = returns[j0:j1, i].sum().item()  # cumulative returns of an episode
                steps_num = j1 - j0  # step number of an episode
                returns_step_list.append((reward_sum, steps_num))

                j0 = j1
    return returns_step_list


def Get_cumulative_rewards_and_step_single_env_parallel(env, actor) -> Tuple[TEN, TEN]:
    import multiprocessing
    from copy import deepcopy

    import numpy as np

    # 1. 准备配置
    num_workers = env.eval_num_workers

    # 4. 准备 Actor (必须转到 CPU !)
    # GPU 模型在多进程 Fork/Spawn 时极易出错且效率低 (Batch=1时)
    actor_cpu = deepcopy(actor).to("cpu")
    actor_cpu.eval()  # 确保是评估模式

    # 5. 准备 Environment
    # 注意：如果 self.env 包含不可序列化的对象（如 C++ 指针、打开的文件句柄），
    # 这里直接传 self.env 会报错。
    # 如果报错，您需要改为传递 env_class 和 env_args 在子进程内重建环境。
    # 这里假设您的 env 是可以 pickle 的。
    def _run_mode(mode: str, total_times: int) -> Tuple[TEN, Union[List[float], None]]:
        # 2. 生成所有任务 ID (假设 ID 是从 0 到 total_times-1)
        all_ids = np.arange(total_times)

        # 3. 将任务 ID 切分为 chunks 分发给 Worker
        chunks = np.array_split(all_ids, num_workers)

        # 5. 准备 Environment (独立拷贝，避免状态干扰)
        env_copy = deepcopy(env)

        # 6. 组装参数
        worker_args = [(actor_cpu, env_copy, chunk, env.max_step, "cpu", mode) for chunk in chunks]

        # 7. 启动并行池
        try:
            ctx = multiprocessing.get_context("fork")
        except:
            ctx = multiprocessing.get_context("spawn")

        with ctx.Pool(processes=num_workers) as pool:
            results_nested = pool.starmap(_eval_worker, worker_args)

        flat_results = []
        action_counts_total = None
        for sub_results, sub_counts in results_nested:
            flat_results.extend(sub_results)
            if sub_counts is not None:
                if action_counts_total is None:
                    action_counts_total = np.asarray(sub_counts, dtype=np.int64)
                else:
                    action_counts_total += np.asarray(sub_counts, dtype=np.int64)

        action_pct = None
        if action_counts_total is not None:
            total = int(action_counts_total.sum())
            if total > 0:
                action_pct = (action_counts_total / total * 100.0).tolist()

        return th.tensor(flat_results, dtype=th.float32), action_pct

    t0 = time.time()
    rewards_steps_ten_eval, eval_action_pct = _run_mode("eval", env.eval_pool_size)
    rewards_steps_ten_test, test_action_pct = _run_mode("test", env.test_pool_size)
    t1 = time.time()
    total_time = t1 - t0
    print(f"| Eval+Test Time Cost: {total_time:.2f} seconds", flush=True)

    return rewards_steps_ten_eval, rewards_steps_ten_test, eval_action_pct, test_action_pct


def draw_learning_curve(recorder: np.ndarray, fig_title: str = "learning_curve", save_path: str = "learning_curve.jpg"):
    steps = recorder[:, 0]  # x-axis is training steps
    r_avg = recorder[:, 1]
    r_std = recorder[:, 2]
    r_exp = recorder[:, 3]
    obj_c = recorder[:, 4]
    obj_a = recorder[:, 5]

    """plot subplots"""
    import matplotlib as mpl

    mpl.use("Agg")
    """Generating matplotlib graphs without a running X server [duplicate]
    write `mpl.use('Agg')` before `import matplotlib.pyplot as plt`
    https://stackoverflow.com/a/4935945/9293137
    """

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2)

    """axs[0]"""
    ax00 = axs[0]
    ax00.cla()

    ax01 = axs[0].twinx()
    color01 = "darkcyan"
    ax01.set_ylabel("Explore AvgReward", color=color01)
    ax01.plot(
        steps,
        r_exp,
        color=color01,
        alpha=0.5,
    )
    ax01.tick_params(axis="y", labelcolor=color01)

    color0 = "lightcoral"
    ax00.set_ylabel("Episode Return", color=color0)
    ax00.plot(steps, r_avg, label="Episode Return", color=color0)
    ax00.fill_between(steps, r_avg - r_std, r_avg + r_std, facecolor=color0, alpha=0.3)
    ax00.grid()
    """axs[1]"""
    ax10 = axs[1]
    ax10.cla()

    ax11 = axs[1].twinx()
    color11 = "darkcyan"
    ax11.set_ylabel("objC", color=color11)
    ax11.fill_between(
        steps,
        obj_c,
        facecolor=color11,
        alpha=0.2,
    )
    ax11.tick_params(axis="y", labelcolor=color11)

    color10 = "royalblue"
    ax10.set_xlabel("Total Steps")
    ax10.set_ylabel("objA", color=color10)
    ax10.plot(steps, obj_a, label="objA", color=color10)
    ax10.tick_params(axis="y", labelcolor=color10)
    for plot_i in range(6, recorder.shape[1]):
        other = recorder[:, plot_i]
        ax10.plot(steps, other, label=f"{plot_i}", color="grey", alpha=0.5)
    ax10.legend()
    ax10.grid()

    """plot save"""
    plt.title(fig_title, y=2.3)
    plt.savefig(save_path)
    plt.close("all")  # avoiding warning about too many open figures, rcParam `figure.max_open_warning`
    # plt.show()  # if use `mpl.use('Agg')` to draw figures without GUI, then plt can't plt.show()


"""learning curve"""


"""def demo_evaluator_actor_pth():
    import gym
    from elegantrl.agents.AgentPPO import AgentPPO
    from elegantrl.train.config import Config, build_env

    gpu_id = 0  # >=0 means GPU ID, -1 means CPU

    agent_class = AgentPPO

    env_class = gym.make
    env_args = {'num_envs': 1,
                'env_name': 'LunarLanderContinuous-v2',
                'max_step': 1000,
                'state_dim': 8,
                'action_dim': 2,
                'if_discrete': False,
                'target_return': 200,

                'id': 'LunarLanderContinuous-v2'}

    # actor_path = './LunarLanderContinuous-v2_PPO_1/actor.pt'
    eval_times = 4
    net_dim = 2 ** 7

    '''init'''
    args = Config(agent_class=agent_class, env_class=env_class, env_args=env_args)
    env = build_env(env_class=args.env_class, env_args=args.env_args)
    act = agent_class(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id, args=args).act
    # act.load_state_dict(th.load(actor_path, map_location=lambda storage, loc: storage))

    '''evaluate'''
    r_s_ary = [get_rewards_and_steps(env, act) for _ in range(eval_times)]
    r_s_ary = np.array(r_s_ary, dtype=np.float32)
    r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step

    print(f'|r_avg {r_avg}  s_avg {s_avg}', flush=True)
    return r_avg, s_avg


def demo_evaluate_actors(dir_path: str, gpu_id: int, agent, env_args: dict, eval_times=2, net_dim=128):
    import gym
    from elegantrl.train.config import build_env
    # dir_path = './LunarLanderContinuous-v2_PPO_1'
    # gpu_id = 0
    # agent_class = AgentPPO
    # net_dim = 2 ** 7

    env_class = gym.make
    # env_args = {'num_envs': 1,
    #             'env_name': 'LunarLanderContinuous-v2',
    #             'max_step': 1000,
    #             'state_dim': 8,
    #             'action_dim': 2,
    #             'if_discrete': False,
    #             'target_return': 200,
    #             'eval_times': 2 ** 4,
    #
    #             'id': 'LunarLanderContinuous-v2'}
    # eval_times = 2 ** 1

    '''init'''
    env = build_env(env_class=env_class, env_args=env_args)
    act = agent(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id).act

    '''evaluate'''
    step_epi_r_s_ary = []

    act_names = [name for name in os.listdir(dir_path) if len(name) == 19]
    for act_name in act_names:
        act_path = f"{dir_path}/{act_name}"

        act.load_state_dict(th.load(act_path, map_location=lambda storage, loc: storage))
        r_s_ary = [get_rewards_and_steps(env, act) for _ in range(eval_times)]
        r_s_ary = np.array(r_s_ary, dtype=np.float32)
        r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step

        step = int(act_name[6:15])

        step_epi_r_s_ary.append((step, r_avg, s_avg))

    step_epi_r_s_ary = np.array(step_epi_r_s_ary, dtype=np.float32)

    '''sort by step'''
    step_epi_r_s_ary = step_epi_r_s_ary[step_epi_r_s_ary[:, 0].argsort()]
    return step_epi_r_s_ary


def demo_load_pendulum_and_render():
    import torch
    from elegantrl.agents.AgentPPO import AgentPPO
    from elegantrl.train.config import Config, build_env

    gpu_id = 0  # >=0 means GPU ID, -1 means CPU

    agent_class = AgentPPO

    from elegantrl.envs.CustomGymEnv import PendulumEnv
    env_class = PendulumEnv
    env_args = {'num_envs': 1,
                'env_name': 'Pendulum-v1',
                'state_dim': 3,
                'action_dim': 1,
                'if_discrete': False, }

    actor_path = './Pendulum-v1_PPO_0/actor.pt'
    net_dim = 2 ** 7

    '''init'''
    env = build_env(env_class=env_class, env_args=env_args)
    args = Config(agent_class=agent_class, env_class=env_class, env_args=env_args)
    act = agent_class(net_dim, env.state_dim, env.action_dim, gpu_id=gpu_id, args=args).act
    act.load_state_dict(torch.load(actor_path, map_location=lambda storage, loc: storage))

    '''evaluate'''
    # eval_times = 2 ** 7
    # from elegantrl.envs.CustomGymEnv import PendulumEnv
    # eval_env = PendulumEnv()
    # from elegantrl.train.evaluator import get_cumulative_returns_and_step
    # r_s_ary = [get_cumulative_returns_and_step(eval_env, act) for _ in range(eval_times)]
    # r_s_ary = np.array(r_s_ary, dtype=np.float32)
    # r_avg, s_avg = r_s_ary.mean(axis=0)  # average of episode return and episode step
    #
    # print(f'|r_avg {r_avg}  s_avg {s_avg}', flush=True)

    '''render'''
    max_step = env.max_step
    if_discrete = env.if_discrete
    device = next(act.parameters()).device  # net.parameters() is a Python generator.

    state = env.reset()
    steps = None
    returns = 0.0  # sum of rewards in an episode
    for steps in range(max_step):
        s_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        a_tensor = act(s_tensor).argmax(dim=1) if if_discrete else act(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because using th.no_grad() outside
        state, reward, done, _ = env.step(action * 2)  # for Pendulum specially
        returns += reward
        env.render()

        if done:
            break
    returns = getattr(env, 'cumulative_returns', returns)
    steps += 1

    print(f"\n| cumulative_returns {returns}"
          f"\n|      episode steps {steps}", flush=True)


def run():
    from elegantrl.agents.AgentPPO import AgentPPO
    flag_id = 1  # int(sys.argv[1])

    gpu_id = [2, 3][flag_id]
    agent = AgentPPO
    env_args = [
        {'num_envs': 1,
         'env_name': 'LunarLanderContinuous-v2',
         'max_step': 1000,
         'state_dim': 8,
         'action_dim': 2,
         'if_discrete': False,
         'target_return': 200,
         'eval_times': 2 ** 4,
         'id': 'LunarLanderContinuous-v2'},

        {'num_envs': 1,
         'env_name': 'BipedalWalker-v3',
         'max_step': 1600,
         'state_dim': 24,
         'action_dim': 4,
         'if_discrete': False,
         'target_return': 300,
         'eval_times': 2 ** 3,
         'id': 'BipedalWalker-v3', },
    ][flag_id]
    env_name = env_args['env_name']

    print('gpu_id', gpu_id, flush=True)
    print('env_name', env_name, flush=True)

    '''save step_epi_r_s_ary'''
    # cwd_path = '.'
    # dir_names = [name for name in os.listdir(cwd_path)
    #              if name.find(env_name) >= 0 and os.path.isdir(name)]
    # for dir_name in dir_names:
    #     dir_path = f"{cwd_path}/{dir_name}"
    #     step_epi_r_s_ary = demo_evaluate_actors(dir_path, gpu_id, agent, env_args)
    #     np.savetxt(f"{dir_path}-step_epi_r_s_ary.txt", step_epi_r_s_ary)

    '''load step_epi_r_s_ary'''
    step_epi_r_s_ary = []

    cwd_path = '.'
    ary_names = [name for name in os.listdir('.')
                 if name.find(env_name) >= 0 and name[-4:] == '.txt']
    for ary_name in ary_names:
        ary_path = f"{cwd_path}/{ary_name}"
        ary = np.loadtxt(ary_path)
        step_epi_r_s_ary.append(ary)
    step_epi_r_s_ary = np.vstack(step_epi_r_s_ary)
    step_epi_r_s_ary = step_epi_r_s_ary[step_epi_r_s_ary[:, 0].argsort()]
    print('step_epi_r_s_ary.shape', step_epi_r_s_ary.shape, flush=True)

    '''plot'''
    import matplotlib.pyplot as plt
    # plt.plot(step_epi_r_s_ary[:, 0], step_epi_r_s_ary[:, 1])

    plot_x_y_up_dw_step = []
    n = 8
    for i in range(0, len(step_epi_r_s_ary), n):
        y_ary = step_epi_r_s_ary[i:i + n, 1]
        if y_ary.shape[0] <= 1:
            continue

        y_avg = y_ary.mean()
        y_up = y_ary[y_ary > y_avg].mean()
        y_dw = y_ary[y_ary <= y_avg].mean()

        y_step = step_epi_r_s_ary[i:i + n, 2].mean()
        x_avg = step_epi_r_s_ary[i:i + n, 0].mean()
        plot_x_y_up_dw_step.append((x_avg, y_avg, y_up, y_dw, y_step))

    if_show_episode_step = True
    color0 = 'royalblue'
    color1 = 'lightcoral'
    # color2 = 'darkcyan'
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    title = f"{env_name}_{agent.__name__}_ElegantRL"

    fig, ax = plt.subplots(1)

    plot_x = [item[0] for item in plot_x_y_up_dw_step]
    plot_y = [item[1] for item in plot_x_y_up_dw_step]
    plot_y_up = [item[2] for item in plot_x_y_up_dw_step]
    plot_y_dw = [item[3] for item in plot_x_y_up_dw_step]
    ax.plot(plot_x, plot_y, label='Episode Return', color=color0)
    ax.fill_between(plot_x, plot_y_up, plot_y_dw, facecolor=color0, alpha=0.3)
    ax.set_ylabel('Episode Return', color=color0)
    ax.tick_params(axis='y', labelcolor=color0)
    ax.grid(True)

    if if_show_episode_step:
        ax_twin = ax.twinx()
        plot_y_step = [item[4] for item in plot_x_y_up_dw_step]
        ax_twin.fill_between(plot_x, 0, plot_y_step, facecolor=color1, alpha=0.3)
        ax_twin.set_ylabel('Episode Step', color=color1)
        ax_twin.tick_params(axis='y', labelcolor=color1)
        ax_twin.set_ylim(0, np.max(plot_y_step) * 2)

    print('title', title, flush=True)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # demo_evaluate_actors()
    run()"""
