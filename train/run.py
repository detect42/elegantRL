import os
import time
import numpy as np
import torch as th
import multiprocessing as mp
from copy import deepcopy
from typing import Any, List, Optional, Tuple, cast, Union
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import numpy.random as rd
from .config import build_env
from .replay_buffer import ReplayBuffer
from .evaluator import Evaluator
from .evaluator import get_rewards_and_steps
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, get_class

if os.name == "nt":  # if is WindowOS (Windows NT)
    """Fix bug about Anaconda in WindowOS
    OMP: Error #15: Initializing libIOmp5md.dll, but found libIOmp5md.dll already initialized.
    """
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

"""train"""


def train_agent(args: DictConfig, if_single_process: bool = False):
    if if_single_process:
        print(f"| train_agent_single_process() with GPU_ID {args.sys.gpu_id}", flush=True)
        train_agent_single_process(args)
    elif len(args.sys.learner_gpu_ids) == 0:
        print(f"| train_agent_multiprocessing() with GPU_ID {args.sys.gpu_id}", flush=True)
        train_agent_multiprocessing(args)
    elif len(args.sys.learner_gpu_ids) != 0:
        print(f"| train_agent_multiprocessing_multi_gpu() with GPU_ID {args.sys.learner_gpu_ids}", flush=True)
        train_agent_multiprocessing_multi_gpu(args)
    else:
        ValueError(f"| run.py train_agent: args.sys.learner_gpu_ids = {args.sys.learner_gpu_ids}")


def train_agent_single_process(args: DictConfig):
    # args.init_before_training()
    th.set_grad_enabled(False)

    """init environment"""
    env_class = get_class(args.env._target_)
    env = build_env(env_class, args.env, args.sys.gpu_id)

    """init agent"""
    agent_class = get_class(args.agent._target_)
    agent = agent_class(args.env.state_dim, args.env.action_dim, gpu_id=args.sys.gpu_id, args=args)
    if args.train.continue_train:
        agent.save_or_load_agent(args.eval.cwd, if_save=False)

    """init agent.last_state"""
    state, info_dict = env.reset()
    if args.env.num_envs == 1:
        assert state.shape == (args.env.state_dim,)
        assert isinstance(state, np.ndarray)
        state = th.tensor(state, dtype=th.float32, device=agent.device).unsqueeze(0)
    else:
        state = state.to(agent.device)
    assert state.shape == (args.env.num_envs, args.env.state_dim)
    assert isinstance(state, th.Tensor)
    agent.last_state = state.detach()

    """init buffer"""
    if args.agent.if_off_policy:
        buffer = ReplayBuffer(
            gpu_id=args.sys.gpu_id,
            num_seqs=args.env.num_envs,
            max_size=args.train.buffer_size,
            state_dim=args.env.state_dim,
            action_dim=1 if args.env.if_discrete else args.env.action_dim,
            if_use_per=args.train.if_use_per,
            if_discrete=args.env.if_discrete,
            args=args,
        )
    else:
        buffer = []

    """init evaluator"""
    eval_env_class = get_class(args.eval.env._target_)
    eval_env_cfg = args.eval.env
    eval_env = build_env(eval_env_class, eval_env_cfg, args.sys.gpu_id)
    evaluator = Evaluator(cwd=args.eval.cwd, env=eval_env, args=args, if_tensorboard=True)

    """train loop"""
    cwd = args.eval.cwd
    break_step = args.train.break_step
    horizon_len = args.train.horizon_len
    if_off_policy = args.agent.if_off_policy
    if_save_buffer = args.eval.if_save_buffer

    if_discrete = env.if_discrete
    show_weight = 1000 / horizon_len / args.env.num_envs / args.sys.num_workers

    def action_to_str(_action_ary):  # TODO PLAN to be elegant
        _show_dict = dict(zip(*np.unique(_action_ary, return_counts=True)))
        _show_str = np.array([int(_show_dict.get(action_key, 0) * show_weight) for action_key in range(env.action_dim)])
        return _show_str

    del args

    if_train = True
    while if_train:
        buffer_items = agent.explore_env(env, horizon_len)
        """buffer_items
        buffer_items = (states, actions,           rewards, undones, unmasks)  # off-policy
        buffer_items = (states, actions, logprobs, rewards, undones, unmasks)  # on-policy

        item.shape == (horizon_len, num_workers * num_envs, ...)
        actions.shape == (horizon_len, num_workers * num_envs, action_dim)  # if_discrete=False
        actions.shape == (horizon_len, num_workers * num_envs)              # if_discrete=True
        """
        if if_off_policy:
            buffer.update(buffer_items)
        else:
            buffer[:] = buffer_items

        if if_discrete:
            show_str = action_to_str(_action_ary=buffer_items[1].data.cpu())
        else:  # TODO PLAN add action_dist
            show_str = ""
        exp_r = buffer_items[2].mean().item()  #! on-policy的时候变成了对logprob的mean 需要if

        th.set_grad_enabled(True)
        logging_dict = agent.update_net(buffer)
        logging_dict = {**logging_dict, "show_str": show_str}
        th.set_grad_enabled(False)

        evaluator.evaluate_and_save(actor=agent.act, steps=horizon_len, exp_r=exp_r, logging_dict=logging_dict)
        if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))

    print(f"| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}", flush=True)

    env.close() if hasattr(env, "close") else None
    evaluator.save_training_curve_jpg()
    agent.save_or_load_agent(cwd, if_save=True)
    if if_save_buffer and hasattr(buffer, "save_or_load_history"):
        buffer.save_or_load_history(cwd, if_save=True)


def train_agent_multiprocessing(args: DictConfig):
    # args.init_before_training()

    """Don't set method='fork' when send tensor in GPU"""
    method = "spawn" if os.name == "nt" else "forkserver"  # os.name == 'nt' means Windows NT operating system (WinOS)
    mp.set_start_method(method=method, force=True)

    """build the Pipe"""
    worker_pipes = [Pipe(duplex=False) for _ in range(args.sys.num_workers)]  # receive, send
    learner_pipe = Pipe(duplex=False)
    evaluator_pipe = Pipe(duplex=True)

    """build Process"""
    learner = Learner(learner_pipe=learner_pipe, worker_pipes=worker_pipes, evaluator_pipe=evaluator_pipe, args=args)
    workers = [
        Worker(worker_pipe=worker_pipe, learner_pipe=learner_pipe, worker_id=worker_id, args=args)
        for worker_id, worker_pipe in enumerate(worker_pipes)
    ]
    evaluator = EvaluatorProc(evaluator_pipe=evaluator_pipe, args=args)

    """start Process with single GPU"""
    process_list = [learner, *workers, evaluator]
    for p in process_list:
        p.start()
    for p in process_list:
        p.join()


def train_agent_multiprocessing_multi_gpu(args: DictConfig):
    # args.init_before_training()

    """Don't set method='fork' when send tensor in GPU"""
    method = "spawn" if os.name == "nt" else "forkserver"  # os.name == 'nt' means Windows NT operating system (WinOS)
    mp.set_start_method(method=method, force=True)

    learners_pipe = [Pipe(duplex=True) for _ in args.sys.learner_gpu_ids]
    process_list_list = []
    for gpu_id in args.sys.learner_gpu_ids:
        args = deepcopy(args)
        args.sys.gpu_id = gpu_id

        """Pipe build"""
        worker_pipes = [Pipe(duplex=False) for _ in range(args.sys.num_workers)]  # receive, send
        learner_pipe = Pipe(duplex=False)
        evaluator_pipe = Pipe(duplex=True)

        """Process build"""
        learner = Learner(
            learner_pipe=learner_pipe,
            worker_pipes=worker_pipes,
            evaluator_pipe=evaluator_pipe,
            learners_pipe=learners_pipe,
            args=args,
        )
        workers = [
            Worker(worker_pipe=worker_pipe, learner_pipe=learner_pipe, worker_id=worker_id, args=args)
            for worker_id, worker_pipe in enumerate(worker_pipes)
        ]
        evaluator = EvaluatorProc(evaluator_pipe=evaluator_pipe, args=args)

        """Process append"""
        process_list = [learner, *workers, evaluator]
        process_list_list.append(process_list)

    """Process start"""
    for process_list in process_list_list:
        for process in process_list:
            process.start()

    """Process join"""
    for process_list in process_list_list:
        for process in process_list:
            process.join()


class Learner(Process):
    def __init__(
        self,
        learner_pipe: tuple[Connection, Connection],
        worker_pipes: List[tuple[Connection, Connection]],
        evaluator_pipe: tuple[Connection, Connection],
        learners_pipe: Optional[List[tuple[Connection, Connection]]],
        args: DictConfig,
    ):
        super().__init__()
        self.recv_pipe = learner_pipe[0]
        self.send_pipes = [worker_pipe[1] for worker_pipe in worker_pipes]
        self.eval_pipe = evaluator_pipe[1]
        self.learners_pipe = learners_pipe
        self.args = args

    def run(self):
        args = self.args
        th.set_grad_enabled(False)

        """COMMUNICATE between Learners: init"""
        learner_id = args.sys.learner_gpu_ids.index(args.sys.gpu_id) if len(args.sys.learner_gpu_ids) > 0 else 0
        num_learners = max(1, len(args.sys.learner_gpu_ids))
        num_communications = num_learners - 1
        if len(args.sys.learner_gpu_ids) >= 2:
            assert isinstance(self.learners_pipe, list)
        elif len(args.sys.learner_gpu_ids) == 0:
            assert self.learners_pipe is None
        elif len(args.sys.learner_gpu_ids) == 1:
            ValueError("| Learner: suggest to set `args.sys.learner_gpu_ids=()` in default")

        """Learner init agent"""
        # agent_class = get_class_from_path(args.agent.agent_name)
        # agent = agent_class(args.env.state_dim, args.env.action_dim, gpu_id=args.sys.gpu_id, args=args)
        agent_class = get_class(args.agent._target_)
        agent = agent_class(
            state_dim=args.env.state_dim,
            action_dim=args.env.action_dim,
            gpu_id=args.sys.gpu_id,
            args=args,
        )
        if args.train.continue_train:
            agent.save_or_load_agent(args.eval.cwd, if_save=False)

        if_off_policy = args.agent.if_off_policy
        """Learner init buffer"""
        if if_off_policy:
            buffer = ReplayBuffer(
                gpu_id=args.sys.gpu_id,
                num_seqs=args.env.num_envs * args.sys.num_workers * num_learners,
                max_size=args.train.buffer_size,
                state_dim=args.env.state_dim,
                action_dim=1 if args.env.if_discrete else args.env.action_dim,
                if_use_per=args.train.if_use_per,
                if_discrete=args.env.if_discrete,
                args=args,
            )
        else:
            buffer = []

        if_discrete = args.env.if_discrete
        if_save_buffer = args.eval.if_save_buffer

        num_workers = args.sys.num_workers
        num_envs = args.env.num_envs
        num_steps = args.train.horizon_len * args.sys.num_workers * args.env.num_envs  #! 这里似乎再乘args.num_envs
        num_seqs = args.env.num_envs * args.sys.num_workers * num_learners
        """	•	args.env.num_envs：每个 Worker 同时跑多少个环境（environment vectorization）
	        •	args.sys.num_workers：每个 Learner 底下启动了多少个 Worker 进程
	        •	num_learners：一共有多少个 Learner（对应多少张 GPU，multi-GPU 模式下的 Learner 数量）"""
        state_dim = args.env.state_dim
        action_dim = args.env.action_dim
        horizon_len = args.train.horizon_len
        cwd = args.eval.cwd
        del args
        agent.last_state = th.empty((num_seqs, state_dim), dtype=th.float32, device=agent.device)

        states = th.zeros((horizon_len, num_seqs, state_dim), dtype=th.float32, device=agent.device)
        actions = (
            th.zeros((horizon_len, num_seqs, action_dim), dtype=th.float32, device=agent.device)
            if not if_discrete
            else th.zeros((horizon_len, num_seqs), dtype=th.int32).to(agent.device)
        )
        rewards = th.zeros((horizon_len, num_seqs), dtype=th.float32, device=agent.device)
        undones = th.zeros((horizon_len, num_seqs), dtype=th.bool, device=agent.device)
        unmasks = th.zeros((horizon_len, num_seqs), dtype=th.bool, device=agent.device)
        if if_off_policy:
            buffer_items_tensor = (states, actions, rewards, undones, unmasks)
        else:
            logprobs = th.zeros((horizon_len, num_seqs), dtype=th.float32, device=agent.device)
            buffer_items_tensor = (states, actions, logprobs, rewards, undones, unmasks)

        accumulated_steps = 0
        if_train = True
        while if_train:
            actor = agent.act
            actor = deepcopy(actor).cpu() if os.name == "nt" else actor  # WindowsNT_OS can only send cpu_tensor

            """Learner send actor to Workers"""
            for send_pipe in self.send_pipes:
                send_pipe.send(actor)
            """Learner receive (buffer_items, last_state) from Workers"""
            for _ in range(num_workers):
                worker_id, buffer_items, last_state = self.recv_pipe.recv()

                buf_i = num_envs * worker_id
                buf_j = num_envs * (worker_id + 1)
                for buffer_item, buffer_tensor in zip(buffer_items, buffer_items_tensor):
                    buffer_tensor[:, buf_i:buf_j] = buffer_item.to(agent.device)
                agent.last_state[buf_i:buf_j] = last_state.to(agent.device)
            del buffer_items, last_state

            """COMMUNICATE between Learners: Learner send actor to other Learners"""
            _buffer_len = num_envs * num_workers
            _buffer_items_tensor = [t[:, :_buffer_len].cpu().detach() for t in buffer_items_tensor]
            for shift_id in range(num_communications):  #! 这里shift_id在循环内部没用？
                _learner_pipe = self.learners_pipe[learner_id][0]
                _learner_pipe.send(_buffer_items_tensor)
            """COMMUNICATE between Learners: Learner receive (buffer_items, last_state) from other Learners"""
            for shift_id in range(num_communications):
                _learner_id = (learner_id + shift_id + 1) % num_learners  # other_learner_id
                _learner_pipe = self.learners_pipe[_learner_id][1]
                _buffer_items_tensor = _learner_pipe.recv()

                _buf_i = num_envs * num_workers * (shift_id + 1)
                _buf_j = num_envs * num_workers * (shift_id + 2)
                for buffer_item, buffer_tensor in zip(_buffer_items_tensor, buffer_items_tensor):
                    buffer_tensor[:, _buf_i:_buf_j] = buffer_item.to(agent.device)

            """Learner update training data to (buffer, agent)"""
            if if_off_policy:
                buffer.update(buffer_items_tensor)
            else:
                buffer[:] = buffer_items_tensor

            if if_discrete:
                # 1. 扁平化 actions 并在 GPU 上转为 long 类型
                flat_actions = buffer_items_tensor[1].view(-1).long()
                # 2. 高效统计频次 (GPU)
                # minlength=action_dim 确保即使某些动作没出现，维度也是固定的
                counts = th.bincount(flat_actions, minlength=action_dim)
                # 3. 计算百分比 (归一化到 100)
                total_samples = flat_actions.numel() + 1e-6
                # 结果例如: [10, 20, 70] (对应 10%, 20%, 70%)
                pcts = (counts.float() * 100 / total_samples).long().tolist()
                # 4. 格式化为带 % 的字符串，并去掉引号使其更像数据
                # 结果例如: [10%, 20%, 70%]
                show_str = str([f"{p}%" for p in pcts]).replace("'", "")
            else:
                show_str = ""
            """Learner update network using training data"""
            th.set_grad_enabled(True)
            logging_dict = agent.update_net(buffer)
            logging_dict = {**logging_dict, "action_show_str": show_str}
            th.set_grad_enabled(False)

            if if_off_policy:
                exp_r = (
                    buffer_items_tensor[2].mean().item()
                )  #! 这里很暴力的直接取mean，是不是应该对每一列（一个轨迹求和）再做mean？但也可能我们假设了所有轨迹都一样长
            else:
                exp_r = buffer_items_tensor[3].mean().item()

            accumulated_steps += num_steps
            """Learner receive training signal from Evaluator"""
            if self.eval_pipe.poll():  # whether there is any data available to be read of this pipe0
                if_train = self.eval_pipe.recv()  # True means evaluator in idle moments.
                self.eval_pipe.send((actor, accumulated_steps, exp_r, logging_dict))
                print(logging_dict)
                accumulated_steps = 0
            else:
                # print("| Learner: Evaluator Pipe No Data Poll()", flush=True)  #
                pass

        """Learner send the terminal signal to workers after break the loop"""
        print("| Learner Close Worker", flush=True)
        for send_pipe in self.send_pipes:
            send_pipe.send(None)
            time.sleep(0.1)

        """save"""
        agent.save_or_load_agent(cwd, if_save=True)
        if if_save_buffer and hasattr(buffer, "save_or_load_history"):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {cwd}", flush=True)
            buffer.save_or_load_history(cwd, if_save=True)
            print(f"| LearnerPipe.run: ReplayBuffer saved  in {cwd}", flush=True)
        print("| Learner Closed", flush=True)


class Worker(Process):
    def __init__(
        self,
        worker_pipe: tuple[Connection, Connection],
        learner_pipe: tuple[Connection, Connection],
        worker_id: int,
        args: DictConfig,
    ):
        super().__init__()
        self.recv_pipe = worker_pipe[0]
        self.send_pipe = learner_pipe[1]
        self.worker_id = worker_id
        self.args = args

    def run(self):
        args = self.args
        worker_id = self.worker_id
        th.set_grad_enabled(False)

        """init environment"""
        env_class = get_class(args.env._target_)
        # print(env_class)
        env = build_env(env_class, args.env, args.sys.gpu_id)

        """init agent"""
        agent_class = get_class(args.agent._target_)
        agent = agent_class(args.env.state_dim, args.env.action_dim, gpu_id=args.sys.gpu_id, args=args)
        if args.train.continue_train:
            agent.save_or_load_agent(args.eval.cwd, if_save=False)

        """init agent.last_state"""
        state, info_dict = env.reset()
        if args.env.num_envs == 1:
            assert state.shape == (args.env.state_dim,)
            assert isinstance(state, np.ndarray)
            state = th.tensor(state, dtype=th.float32, device=agent.device).unsqueeze(0)
        else:
            assert state.shape == (args.env.num_envs, args.env.state_dim)
            assert isinstance(state, th.Tensor)
            state = state.to(agent.device)
        assert state.shape == (args.env.num_envs, args.env.state_dim)
        assert isinstance(state, th.Tensor)
        agent.last_state = state.detach()

        """init buffer"""
        horizon_len = args.train.horizon_len

        """loop"""
        del args
        # import time  #!
        th.set_num_threads(1)
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1, user_api="blas"):
            while True:
                """Worker receive actor from Learner"""
                actor = self.recv_pipe.recv()
                if actor is None:
                    break
                agent.act = (
                    actor.to(agent.device) if os.name == "nt" else actor
                )  # WindowsNT_OS can only send cpu_tensor
                # t0 = time.time()  #!
                """Worker send the training data to Learner"""
                buffer_items = agent.explore_env(env, horizon_len)
                last_state = agent.last_state
                if os.name == "nt":  # WindowsNT_OS can only send cpu_tensor
                    buffer_items = [t.cpu() for t in buffer_items]
                    last_state = deepcopy(last_state).cpu()
                self.send_pipe.send((worker_id, buffer_items, last_state))
                # t1 = time.time()  #!
                # print(f"| Worker-{worker_id} Explore Time: {t1 - t0:.3f}s", flush=True)  #!

        env.close() if hasattr(env, "close") else None
        print(f"| Worker-{self.worker_id} Closed", flush=True)


class EvaluatorProc(Process):
    def __init__(self, evaluator_pipe: tuple[Connection, Connection], args: DictConfig):
        super().__init__()
        self.pipe0 = evaluator_pipe[0]
        self.pipe1 = evaluator_pipe[1]
        self.args = args

    def run(self):
        args = self.args
        th.set_grad_enabled(False)

        """init evaluator"""
        eval_env_class = get_class(args.eval.env._target_)
        eval_env_cfg = args.eval.env
        eval_env = build_env(eval_env_class, eval_env_cfg, args.sys.gpu_id)
        evaluator = Evaluator(cwd=args.eval.cwd, env=eval_env, args=args, if_tensorboard=True)

        """loop"""
        cwd = args.eval.cwd
        break_step = args.train.break_step
        device = th.device(f"cuda:{args.sys.gpu_id}" if (th.cuda.is_available() and (args.sys.gpu_id >= 0)) else "cpu")
        del args

        if_train = True
        self.pipe0.send(if_train)
        while if_train:
            # print("| Evaluator: Waiting for Learner", flush=True)
            """Evaluator receive training log from Learner"""
            actor, steps, exp_r, logging_dict = self.pipe0.recv()
            """Evaluator evaluate the actor and save the training log"""
            if actor is None:
                evaluator.total_step += steps  # update total_step but don't update recorder
            else:
                actor = actor.to(device) if os.name == "nt" else actor  # WindowsNT_OS can only send cpu_tensor
                evaluator.evaluate_and_save(actor=actor, steps=steps, exp_r=exp_r, logging_dict=logging_dict)

            """Evaluator send the training signal to Learner"""
            if_train = (evaluator.total_step <= break_step) and (not os.path.exists(f"{cwd}/stop"))
            self.pipe0.send(if_train)

        """Evaluator save the training log and draw the learning curve"""
        evaluator.save_training_curve_jpg()
        print(f"| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}", flush=True)

        print("| Evaluator Closing", flush=True)
        while self.pipe1.poll():  # whether there is any data available to be read of this pipe
            while self.pipe0.poll():
                try:
                    self.pipe0.recv()
                except RuntimeError:
                    print("| Evaluator Ignore RuntimeError in self.pipe0.recv()", flush=True)
                time.sleep(1)
            time.sleep(1)

        eval_env.close() if hasattr(eval_env, "close") else None
        print("| Evaluator Closed", flush=True)


"""render"""


def valid_agent(
    env_class, env_cfg: DictConfig, net_dims: List[int], agent_class, actor_path: str, render_times: int = 8
):
    env = build_env(env_class, env_cfg)

    state_dim = env_cfg.state_dim
    action_dim = env_cfg.action_dim
    agent = agent_class(state_dim, action_dim, gpu_id=-1)
    actor = agent.act

    print(f"| render and load actor from: {actor_path}", flush=True)
    actor.load_state_dict(th.load(actor_path, map_location=lambda storage, loc: storage))
    for i in range(render_times):
        cumulative_reward, episode_step = get_rewards_and_steps(env, actor, if_render=True)
        print(f"|{i:4}  cumulative_reward {cumulative_reward:9.3f}  episode_step {episode_step:5.0f}", flush=True)


def render_agent(
    env_class, env_cfg: DictConfig, net_dims: List[int], agent_class, actor_path: str, render_times: int = 8
):
    env = build_env(env_class, env_cfg)

    state_dim = env_cfg.state_dim
    action_dim = env_cfg.action_dim
    agent = agent_class(state_dim, action_dim, gpu_id=-1)
    actor = agent.act
    del agent

    print(f"| render and load actor from: {actor_path}", flush=True)
    actor.load_state_dict(th.load(actor_path, map_location=lambda storage, loc: storage))
    for i in range(render_times):
        cumulative_reward, episode_step = get_rewards_and_steps(env, actor, if_render=True)
        print(f"|{i:4}  cumulative_reward {cumulative_reward:9.3f}  episode_step {episode_step:5.0f}", flush=True)
