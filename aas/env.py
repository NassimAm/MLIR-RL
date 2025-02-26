from aas import config as cfg
from aas.observation.benchmark import BenchmarkFeatures, extract_bench_features_from_file, extract_bench_features_from_code
from aas.observation.operation import extract_op_features_from_code
from aas.transforms import transform_dialect_img2col
from aas.state import OperationState
from aas.agent import AlphaAutoScheduler
from aas.wrappers import AASNetworkEstimation
from aas.evaluation import evaluate_code_with_timeout, evaluate_benchmark_code_with_timeout
from typing import Optional, Literal
import random
import string
import os
import json
from tqdm import tqdm
from collections import deque
import math
from utils.log import print_info, print_alert, print_success, print_error
import neptune
import torch
from concurrent.futures import ProcessPoolExecutor
import time
import psutil


class AASOpEnv:
    """Environment for training an AlphaAutoScheduler agent to optimize code by finding the best transformation list operation-wise."""

    benchmarks_data: list[BenchmarkFeatures]
    """List that contains all benchmarks features."""
    tmp_file_path: str
    """The temporary file to store the intermediate representations."""

    def __init__(self, tmp_file_path: Optional[str] = None):
        """Initialize the environment.

        Args:
            tmp_file_path (Optional[str]): The temporary file to store the intermediate representations. Defaults to None.
        """
        # Generate a random file to be used in order to apply the transformations and evaluate the code
        # This is done in order to enable having multiple experiments at the same time, by letting each
        # experiment use a separate unique file to read and write intermediate representations
        if tmp_file_path is None:
            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            tmp_file_path = os.path.join('tmp', f'{random_str}.mlir')
        with open(tmp_file_path, "w") as file:
            file.write("")
        self.tmp_file_path = tmp_file_path

        # Get execution database path or generate a new one
        if not cfg.exec_db_path:
            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            cfg.exec_db_path = os.path.join('tmp', f'{random_str}.json')
            with open(cfg.exec_db_path, "w") as file:
                file.write("{}")

        # Get benchmarks data
        self.benchmarks_data = []
        if cfg.data_format == "mlir":
            # Load execution times from json file
            with open(cfg.json_file, "r") as file:
                benchmarks_json: dict[str, float] = json.load(file)
            # Build benchmark features
            for bench_name, exec_time in benchmarks_json.items():
                bench_file = os.path.join(cfg.benchmarks_folder_path, bench_name + ".mlir")
                benchmark_data = extract_bench_features_from_file(bench_name, bench_file, exec_time)
                self.benchmarks_data.append(benchmark_data)
        else:
            # Load operations data from json file
            with open(cfg.json_file, "r") as file:
                json_data = json.load(file)
            operation_filter = [
                'linalg.matmul',
                'linalg.conv_2d',
                # 'pooling',
                # 'generic',
                'linalg.add',
            ]
            json_data = {op: details for op, details in json_data.items() if any([s in op for s in operation_filter])}
            json_data = [(op, details) for op, details in json_data.items()]

            # Get the AST of the MLIR code and give a tag to each linalg operation
            # The last operation represents the operations that we want to optimize (the first operations are just linalg.fills)
            for i in tqdm(range(len(json_data)), desc='Loading benchmarks'):
                # Get full MLIR code and execution time
                code = json_data[i][1]["transform_wrapped_operation"]
                exec_time = json_data[i][1]["execution_time"]
                # Build benchmark features
                bench_name = json_data[i][0]
                benchmark_data = extract_bench_features_from_code(bench_name, code, exec_time)
                # Apply Img2Col transformation to conv_2d operations
                for op_tag, op_features in benchmark_data.operations.items():
                    if op_features.operation_type == 'conv_2d':
                        new_code = transform_dialect_img2col(benchmark_data.code, op_tag, self.tmp_file_path)
                        new_operation_features = extract_op_features_from_code(new_code, op_tag)
                        new_operation_features.operation_type = 'conv_2d+img2col'
                        benchmark_data.operations[op_tag] = new_operation_features
                        benchmark_data.code = new_code
                self.benchmarks_data.append(benchmark_data)

    def reset(self, idx: Optional[int] = None, mode: Literal['random', 'sequential'] = 'random'):
        """Reset the environment.

        Args:
            idx (Optional[int]): The index of the benchmark to set the environement to. If None, a random benchmark is selected. Defaults to None.
            mode (Literal['random', 'sequential']): The mode to select the next benchmark. Defaults to 'random'.

        Returns:
            OperationState: The initial state of the environment.
        """
        if idx is not None:
            # We get the benchmark with the right index
            self.bench_index = idx
        else:
            if mode == 'sequential':
                # Get the next benchmark
                self.bench_index = (self.bench_index + 1) % len(self.benchmarks_data)
            else:
                # Get a random benchmark
                self.bench_index = random.randint(0, len(self.benchmarks_data) - 1)

        # Get benchmark data
        benchmark_data = self.benchmarks_data[self.bench_index]

        # Get the last operation
        operation_tag = benchmark_data.operation_tags[-1]
        operation_features = benchmark_data.operations[operation_tag]

        # Build the initial state
        state = OperationState(
            bench_features=benchmark_data,
            operation_tag=operation_tag,
            operation_features=operation_features,
            step_count=0,
            transformation_history=[]
        )

        # Return that state
        return state

    def step(self, state: OperationState, mode: Literal['random', 'sequential'] = 'random'):
        """Take a step in the environment given an agent.

        Args:
            state (OperationState): The current state of the environment.
            mode (Literal['random', 'sequential']): The mode to select the next operation. Defaults to 'random'.

        Returns:
            OperationState: The next state of the environment.
            bool: Whether the benchmark optimization is over or not.
        """

        # Get benchmark data
        bench_data = self.benchmarks_data[self.bench_index]

        # Indicates that the benchmark optimization is over or not
        terminated = True
        if cfg.optimization_mode == "all":
            op_index = bench_data.operation_tags.index(state.operation_tag)
            if op_index > 0:
                # Benchmark optimization is not over
                terminated = False
                # Build a new state that points to the next operation
                new_op_tag = bench_data.operation_tags[op_index - 1]
                new_op_features = bench_data.operations[new_op_tag]
                next_state = OperationState(
                    bench_features=bench_data,
                    operation_tag=new_op_tag,
                    operation_features=new_op_features,
                    step_count=0,
                    transformation_history=[]
                )

        # If the benchmark optimization is over, we reset the environment
        if terminated:
            next_state = self.reset(mode=mode)

        return next_state, terminated

    def get_reward(self, state: OperationState, exec_time: int):
        """Get the reward for the agent given the execution time of the optimized code.

        Args:
            state (OperationState): The current state of the environment.
            exec_time (int): The execution time of the optimized code.

        Returns:
            float: The reward for the agent.
        """
        root_exec_time = state.bench_features.root_exec_time
        speedup = root_exec_time / exec_time if exec_time > 0 and root_exec_time > 0 else 1.0
        return math.log2(speedup)

    def compare(self, inputs: list[tuple[OperationState]], agent1: AlphaAutoScheduler, agent2: AlphaAutoScheduler):
        """Compare two agents on a set of benchmarks.

        Args:
            inputs (list[tuple[OperationState]]): The states to compare the agents on.
            agent1 (AlphaAutoScheduler): The first agent to compare.
            agent2 (AlphaAutoScheduler): The second agent to compare.

        Returns:
            float: The score of the comparison.
            list[float]: The speedups of the first agent.
            list[float]: The speedups of the second agent.
        """
        # Run specified number of full benchmark episodes
        score = 0
        speedups1: list[float] = []
        speedups2: list[float] = []
        ins: list[tuple[OperationState, str]] = []
        terminated_list: list[bool] = []
        mcts_start_time = time.time()
        # Gather states
        for (state,) in inputs:
            # Add the current state to the list of states
            ins.append((state, 'greedy'))
            terminated_list.append(state.is_final())
        # Set number of torch threads to 1 to avoid issues with multiprocessing
        original_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        with ProcessPoolExecutor() as executor:
            # Run the agent1 on the current states
            trajectories1 = list(executor.map(agent1.run_parallel, ins))
        with ProcessPoolExecutor() as executor:
            # Run the agent2 on the current states
            trajectories2 = list(executor.map(agent2.run_parallel, ins))
        # Reset the number of torch threads
        torch.set_num_threads(original_num_threads)
        # Get last states
        bench_states1: list[list[OperationState]] = []
        bench_states2: list[list[OperationState]] = []
        tmp_states1 = []
        tmp_states2 = []
        for trajectory1, trajectory2, terminated in zip(trajectories1, trajectories2, terminated_list):
            tmp_states1.append(trajectory1[-1][0])
            tmp_states2.append(trajectory2[-1][0])
            if terminated:
                bench_states1.append(tmp_states1)
                bench_states2.append(tmp_states2)
                tmp_states1 = []
                tmp_states2 = []
        # Get MCTS search time
        mcts_end_time = time.time()
        print_info(f"MCTS Search ended in {mcts_end_time - mcts_start_time} s ...")
        # Execute optimized operations
        for i in tqdm(range(len(bench_states1)), desc="Evaluation execution"):
            last_states1 = bench_states1[i]
            last_states2 = bench_states2[i]
            bench_features = last_states1[0].bench_features
            # Evaluate the transformed code
            exec_time1, assertion1, _ = evaluate_benchmark_code_with_timeout(last_states1, self.tmp_file_path)
            exec_time2, assertion2, _ = evaluate_benchmark_code_with_timeout(last_states2, self.tmp_file_path)
            # Calculate speedups
            root_exec_time = bench_features.root_exec_time
            speedup1 = root_exec_time / exec_time1 if exec_time1 is not None and assertion1 else 1.0
            speedup2 = (root_exec_time / exec_time2 if exec_time2 is not None and assertion2 else 1.0)
            print_info(f"Speedup1: {speedup1}, Speedup2: {speedup2}")
            # Calculate score based on ratio between speedups (positive score = keep new agent, negative score = keep old agent)
            score += (speedup1 / speedup2 - 1.0) if speedup1 >= speedup2 else -(speedup2 / speedup1 - 1.0)
            speedups1.append(speedup1)
            speedups2.append(speedup2)
        # Return the score
        return score, speedups1, speedups2

    def eval(self, agent: AlphaAutoScheduler, mode: Literal['greedy', 'stochastic'] = 'greedy'):
        """Evaluate the agent on a set of benchmarks.

        Args:
            agent (AlphaAutoScheduler): The agent to evaluate.

        Returns:
            list[str]: The names of the benchmarks.
            list[float]: The speedups reached by the agent.
        """
        # Run specified number of full benchmark episodes
        bench_names = []
        speedups = []
        state = self.reset()
        for _ in tqdm(range(len(self.benchmarks_data)), desc="Evaluation"):
            optimized_states: list[OperationState] = []
            terminated = False
            while not terminated:
                # Run the agent on the current state
                optimized_state = agent.eval(state, mode=mode)
                # Save the last state in trajectory
                optimized_states.append(optimized_state)
                # Take a step in the environment
                state, terminated = self.step(state, mode='sequential')
            # Evaluate the transformed code
            exec_time, assertion, _ = evaluate_benchmark_code_with_timeout(optimized_states, self.tmp_file_path)
            # Calculate speedup
            bench_features = optimized_states[0].bench_features
            root_exec_time = bench_features.root_exec_time
            if exec_time is not None and assertion:
                speedup = root_exec_time / exec_time
                bench_names.append(bench_features.bench_name)
                speedups.append(speedup)
        # Return the speedups
        return bench_names, speedups


class AASTrainer:
    """Trainer for an AlphaAutoScheduler agent."""

    train_env: AASOpEnv
    """The environment to train the agent on."""
    eval_env: AASOpEnv
    """The environment to evaluate the agent on."""
    agent: AlphaAutoScheduler
    """The agent agent to train."""
    prev_agent: Optional[AlphaAutoScheduler]
    """The previous agent to compare with the current one."""
    data: deque[tuple[OperationState, AASNetworkEstimation]]
    """The data queue to store training data."""
    save_file_path: str
    """The path to save the agent to."""

    def __init__(self, env_type: Literal["op"] = "op", save_file_path: Optional[str] = None):
        """Initialize the trainer.

        Args:
            env (AASOpEnv): The environment to train the agent on.
            save_file_path (Optional[str]): The path to save the agent to. Defaults to None.
        """
        # Initialize the environment
        if env_type == "op":
            self.train_env = AASOpEnv()
            self.eval_env = AASOpEnv()
        else:
            raise ValueError(f"Invalid environment type ({env_type}). Please choose 'op' for operation-wise optimization.")
        # Initialize agents
        if save_file_path is None:
            self.save_file_path = os.path.join("models", "aas_agent.pt")
        else:
            self.save_file_path = save_file_path
        self.agent = AlphaAutoScheduler(self.train_env.get_reward)
        self.agent.save(self.save_file_path)
        self.prev_agent = None
        # Initialize data queue
        self.data = deque(maxlen=cfg.data_queue_max_length)

    def train(self, neptune_logs: Optional[neptune.Run] = None):
        """Train the agent to optimize benchmarks."""
        # Initialize the environment
        prev_eval_speedups: Optional[list] = None
        prev_eval_bench_names: Optional[list[str]] = None
        agent_eval_changed = False
        # Get training states
        ins: list[tuple[OperationState]] = []
        state = self.train_env.reset()
        for _ in range(cfg.nb_train_eps):
            # Add the current state to the training states
            ins.append((state,))
            # Take a step in the environment
            state, _ = self.train_env.step(state)
        # Loop trough iterations
        for i in tqdm(range(cfg.nb_iterations), desc="Main Loop"):
            # Set agent temperature
            self.agent.set_temperature(1.0 if i < 500 else 0.5 if i < 750 else 0.25)
            # ======================== Gather training data ========================
            print_info("Started MCTS search ...")
            print_error("Number of open files:", len(psutil.Process().open_files()))
            mcts_start_time = time.time()
            # Set number of torch threads to 1 to avoid issues with multiprocessing
            original_num_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            with ProcessPoolExecutor() as executor:
                # Run the agent on the current state
                trajectories = list(executor.map(self.agent.run_parallel, ins))
            # Reset the number of torch threads
            torch.set_num_threads(original_num_threads)
            mcts_end_time = time.time()
            print_info("Number of trajectories:", len(trajectories))
            print_info("Total number of data points:", sum(len(trajectory) for trajectory in trajectories))
            print_info(f"MCTS search ended in {mcts_end_time - mcts_start_time} s ...")

            # ======================== Execute trajectories ========================
            print_info("Started execution ...")
            print_error("Number of open files:", len(psutil.Process().open_files()))
            speedups = []
            last_states: list[OperationState] = []
            len_train_eps = 0
            for j in tqdm(range(len(trajectories)), desc='Trajectories Execution'):
                # Get the last state of the trajectory
                last_state, _ = trajectories[j][-1]
                len_train_eps += len(last_state.transformation_history)
                # Run the code with last state transformation list
                exec_time, assertion, transformed_code = evaluate_code_with_timeout(last_state, self.train_env.tmp_file_path)
                # Get the reward
                value = self.train_env.get_reward(last_state, exec_time)
                # Add the trajectory to the data queue
                if exec_time is not None and assertion:
                    speedups.append(last_state.bench_features.root_exec_time / exec_time)
                    last_states.append(last_state)
                    for trajectory_state, trajectory_policy in trajectories[j]:
                        self.data.append((trajectory_state, AASNetworkEstimation(
                            policy=trajectory_policy,
                            value=value
                        )))
                else:
                    if exec_time is None:
                        if transformed_code:
                            print_error(f"EXECUTION ERROR: ({last_state.bench_features.bench_name} {last_state.operation_tag})")
                            print_error("ACTIONS:", last_state.transformation_history)
                        else:
                            print_error(f"TRANSFORMATION ERROR: ({last_state.bench_features.bench_name} {last_state.operation_tag})")
                            print_error("ACTIONS:", last_state.transformation_history)
                    else:
                        print_error(f"ASSERTION FAILED: ({last_state.bench_features.bench_name} {last_state.operation_tag})")
                        print_error("ACTIONS:", last_state.transformation_history)
            if neptune_logs is not None and len(speedups) > 0:
                neptune_logs['train/final_speedup'].extend(speedups, wait=True)
            print_info("Average speedup:", sum(speedups) / len(speedups))
            print_info("Max speedup:", max(speedups))
            max_speedup_state = last_states[speedups.index(max(speedups))]
            print_info("Max speedup schedule:", max_speedup_state.bench_features.bench_name, max_speedup_state.transformation_history)
            print_info("Execution ended ...")

            # ======================== Train the agent =============================
            # Load the previous agent
            self.prev_agent = self.agent.copy()
            # Train the current agent
            print_info("Started training ...")
            print_error("Number of open files:", len(psutil.Process().open_files()))
            train_stats = self.agent.train(list(self.data))
            if neptune_logs is not None:
                neptune_logs['train/selection_loss'].extend(train_stats.selection_loss, wait=True)
                neptune_logs['train/value_loss'].extend(train_stats.value_loss, wait=True)
                for j in range(cfg.max_num_loops):
                    neptune_logs[f'train/parallel_params_loss_{j}'].extend(train_stats.parallel_params_loss[j], wait=True)
            print_info("Training ended ...")

            # ======================= Get training data ============================
            # Get training states
            ins.clear()
            state = self.train_env.reset()
            for _ in range(cfg.nb_train_eps):
                # Add the current state to the training states
                ins.append((state,))
                # Take a step in the environment
                state, _ = self.train_env.step(state)

            # ========== Evaluate the agent with MCTS for next training ============
            # Compare between the current agent and the previous one
            print_info("Started comparison ...")
            print_error("Number of open files:", len(psutil.Process().open_files()))
            score, speedups1, speedups2 = self.train_env.compare(ins, self.agent, self.prev_agent)
            print_info("Score:", score)
            # If the agent did not improve, load the previous agent
            if score < 0:
                # Load the previous agent
                self.agent = self.prev_agent
                print_alert("Agent did not improve, loading the previous agent ...")
                if neptune_logs is not None and len(speedups2) > 0:
                    neptune_logs['eval/comp/final_speedup'].extend(speedups2, wait=True)
            else:
                # Keep the current agent
                agent_eval_changed = True
                print_success("Agent improved, keeping the current agent ...")
                if neptune_logs is not None and len(speedups1) > 0:
                    neptune_logs['eval/comp/final_speedup'].extend(speedups1, wait=True)
            print_info("Comparison ended ...")

            # ================ Evaluate the agent without MCTS =====================
            if i % 5 == 0:
                print_info("Started evaluation ...")
                print_error("Number of open files:", len(psutil.Process().open_files()))
                if agent_eval_changed or prev_eval_speedups is None or prev_eval_bench_names is None:
                    bench_names, greedy_speedups = self.eval_env.eval(self.agent, mode='greedy')
                    prev_eval_bench_names = bench_names
                    prev_eval_speedups = greedy_speedups
                avg_greedy_speedup = sum(prev_eval_speedups) / len(prev_eval_speedups) if len(prev_eval_speedups) > 0 else 0.0
                print_info("Greedy speedups average:", avg_greedy_speedup)
                if neptune_logs is not None:
                    if len(prev_eval_speedups) > 0:
                        neptune_logs['eval/final_speedup'].extend(prev_eval_speedups, wait=True)
                        neptune_logs['eval/average_speedup'].append(avg_greedy_speedup, wait=True)
                        for bench_name, speedup in zip(prev_eval_bench_names, prev_eval_speedups):
                            neptune_logs[f'eval/{bench_name}'].append(speedup, wait=True)
                agent_eval_changed = False
                print_info("Evaluation ended ...")

            # ====================== Save the best agent ===========================
            # Save the best agent so far
            self.agent.save(self.save_file_path)

    def get_best_agent(self):
        """Load the best agent so far.

        Returns:
            AlphaAutoScheduler: The best agent so far.
        """
        return AlphaAutoScheduler.load_from_file(self.save_file_path, self.train_env.get_reward)
