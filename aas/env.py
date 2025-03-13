import multiprocessing.managers
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
import multiprocessing
import time

# To avoid leaking file descriptors when using multiprocessing
# https://github.com/pytorch/pytorch/issues/973#issuecomment-604473515
torch.multiprocessing.set_sharing_strategy('file_system')
# Set the number of threads for torch
torch.set_num_threads(4)


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

    def eval(self, agent: AlphaAutoScheduler, mode: Literal['greedy', 'stochastic'] = 'greedy'):
        """Evaluate the agent on a set of benchmarks.

        Args:
            agent (AlphaAutoScheduler): The agent to evaluate.

        Returns:
            list[float]: The speedups reached by the agent.
        """
        # Run specified number of full benchmark episodes
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
                speedups.append(speedup)
        # Return the speedups
        return speedups


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
    train_output_list: multiprocessing.managers.ListProxy
    """The list to store the output of the parallel MCTS searches."""

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
        self.prev_agent = self.agent.copy()
        # Initialize data queue
        self.data = deque(maxlen=cfg.data_queue_max_length)
        # Initialize pipes for parallel MCTS searches
        manager = multiprocessing.Manager()
        self.train_output_list = manager.list()

    def train(self, neptune_logs: Optional[neptune.Run] = None):
        """Train the agent to optimize benchmarks."""
        # Initialize the environment
        prev_eval_speedups: Optional[list] = None
        exec_db: Optional[dict] = None
        train_state = self.train_env.reset()
        # Loop trough iterations
        for i in tqdm(range(cfg.nb_iterations), desc="Main Loop"):
            # Set agent temperature
            self.agent.set_temperature(1.0 if i < 500 else 0.5 if i < 750 else 0.25)
            # ======================== Gather training data ========================
            # Read execution database
            if cfg.exec_db_path:
                with open(cfg.exec_db_path, "r") as f:
                    exec_db = json.load(f)
            # Start MCTS searches
            print_info("Started MCTS search ...")
            mcts_start_time = time.time()
            # Clear the output list
            self.train_output_list[:] = []
            # Set number of torch threads to 1 to avoid issues with multiprocessing
            original_num_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            processes: list[multiprocessing.Process] = []
            # Run MCTS searches in parallel
            for id in range(cfg.nb_train_eps):
                # Run the agent on the current state
                process = multiprocessing.Process(target=self.agent.run_parallel, args=(id, self.train_output_list, (train_state, exec_db.get(train_state.bench_features.bench_name))))
                process.start()
                processes.append(process)
                train_state, _ = self.train_env.step(train_state)
            # Wait for all processes to finish
            for process in processes:
                process.join()
                process.close()
            # Reset the number of torch threads
            torch.set_num_threads(original_num_threads)
            trajectories: list[list[tuple[OperationState, AASNetworkEstimation]]] = [trajectory for _, trajectory in sorted(list(self.train_output_list), key=lambda x: x[0])]
            mcts_end_time = time.time()
            print_info("Number of trajectories:", len(trajectories))
            print_info("Total number of data points:", sum(len(trajectory) for trajectory in trajectories))
            print_info(f"MCTS search ended in {mcts_end_time - mcts_start_time} s ...")

            # ======================== Execute trajectories ========================
            print_info("Started execution ...")
            speedups = []
            for j in tqdm(range(len(trajectories)), desc='Trajectories Execution'):
                # Get the last state of the trajectory
                last_state, _ = trajectories[j][-1]
                # Run the code with last state transformation list
                exec_time, assertion, transformed_code = evaluate_code_with_timeout(last_state, self.train_env.tmp_file_path)
                # Add the trajectory to the data queue
                if exec_time is not None and assertion:
                    # Get the reward
                    value = self.train_env.get_reward(last_state, exec_time)
                    speedups.append(last_state.bench_features.root_exec_time / exec_time)
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
            print_info("Execution ended ...")

            # ======================== Train the agent =============================
            # Train the current agent
            print_info("Started training ...")
            train_stats = self.agent.train(list(self.data))
            if neptune_logs is not None:
                neptune_logs['train/selection_loss'].extend(train_stats.selection_loss, wait=True)
                neptune_logs['train/value_loss'].extend(train_stats.value_loss, wait=True)
                for j in range(cfg.max_num_loops):
                    neptune_logs[f'train/parallel_params_loss_{j}'].extend(train_stats.parallel_params_loss[j], wait=True)
            print_info("Training ended ...")

            # ================ Evaluate the agent without MCTS =====================
            if i % 5 == 0:
                print_info("Started evaluation ...")
                eval_speedups = self.eval_env.eval(self.agent, mode='greedy')
                avg_eval_speedup = sum(eval_speedups) / len(eval_speedups) if len(eval_speedups) > 0 else 0.0
                print_info("Greedy speedups average:", avg_eval_speedup)
                if neptune_logs is not None:
                    if len(eval_speedups) > 0:
                        neptune_logs['eval/final_speedup'].extend(eval_speedups, wait=True)
                        neptune_logs['eval/average_speedup'].append(avg_eval_speedup, wait=True)
                        bench_names = [bench.bench_name for bench in self.eval_env.benchmarks_data]
                        for bench_name, eval_speedup in zip(bench_names, eval_speedups):
                            neptune_logs[f'eval/{bench_name}'].append(eval_speedup, wait=True)
                if cfg.pitting:
                    score = self.pit(eval_speedups, prev_eval_speedups)
                    if score >= 0:
                        print_success("Agent improved over the previous one with a score of", score)
                        prev_eval_speedups = eval_speedups
                        self.prev_agent = self.agent.copy()
                    else:
                        print_alert("Agent did not improve over the previous one with a score of", score)
                        self.agent = self.prev_agent
                print_info("Evaluation ended ...")

            # ====================== Save the best agent ===========================
            # Save the best agent so far
            self.agent.save(self.save_file_path)

    def pit(self, speedups1: list[float], speedups2: list[float]):
        """Compare two agents on a set of benchmarks.

        Args:
            speedups1 (list[float]): The speedups of the first agent.
            speedups2 (list[float]): The speedups of the second agent.

        Returns:
            float: The score of the comparison.
        """
        # Initialize the score
        score = 0
        # Check if speedups are available
        if not speedups1 or not speedups2:
            return score
        # Calculate score based on ratio between speedups (positive score = keep new agent, negative score = keep old agent)
        for speedup1, speedup2 in zip(speedups1, speedups2):
            score += (speedup1 / speedup2 - 1.0) if speedup1 >= speedup2 else -(speedup2 / speedup1 - 1.0)
        # Return the score
        return score

    def get_best_agent(self):
        """Load the best agent so far.

        Returns:
            AlphaAutoScheduler: The best agent so far.
        """
        return AlphaAutoScheduler.load_from_file(self.save_file_path, self.train_env.get_reward)
