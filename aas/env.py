from aas import config as cfg
from aas.observation.benchmark import BenchmarkFeatures, extract_bench_features_from_file, extract_bench_features_from_code
from aas.state import OperationState
from aas.agent import AlphaAutoScheduler
from aas.wrappers import AASNetworkPolicyEstimation, AASNetworkEstimation
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
            json_data = [(details['operation'], details) for _, details in json_data.items()]

            # Get the AST of the MLIR code and give a tag to each linalg operation
            # The last operation represents the operations that we want to optimize (the first operations are just linalg.fills)
            for i in tqdm(range(len(json_data))):
                # Get full MLIR code and execution time
                code = json_data[i][1]["transform_wrapped_operation"]
                exec_time = json_data[i][1]["execution_time"]
                # Build benchmark features
                bench_name = f"bench_{i}"
                benchmark_data = extract_bench_features_from_code(bench_name, code, exec_time)
                self.benchmarks_data.append(benchmark_data)

    def reset(self, idx: Optional[int] = None):
        """Reset the environment.

        Args:
            idx (Optional[int]): The index of the benchmark to set the environement to. If None, a random benchmark is selected. Defaults to None.

        Returns:
            OperationState: The initial state of the environment.
        """
        if idx is not None:
            # We get the benchmark with the right index
            self.bench_index = idx
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

    def step(self, state: OperationState):
        """Take a step in the environment given an agent.

        Args:
            state (OperationState): The current state of the environment.

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
            next_state = self.reset()

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
        speedup = root_exec_time / exec_time
        return max(-3.0, min(3.0, math.log2(speedup) / 2 - 1.0))
        # best_speedup = max(2.0, bench_features.best_speedup[state.operation_tag])
        # return max(-1.0, min(1.0, math.log2(speedup / best_speedup)))

    def compare(self, agent1: AlphaAutoScheduler, agent2: AlphaAutoScheduler):
        """Compare two agents on a set of benchmarks.

        Args:
            agent1 (AlphaAutoScheduler): The first agent to compare.
            agent2 (AlphaAutoScheduler): The second agent to compare.

        Returns:
            float: The score of the comparison.
            list[float]: The speedups of the first agent.
            list[float]: The speedups of the second agent.
        """
        # Run specified number of full benchmark episodes
        score = 0
        speedups1 = []
        speedups2 = []
        manager = multiprocessing.Manager()
        processes: list[multiprocessing.Process] = []
        terminated_list: list[bool] = []
        trajectories_list = manager.list()
        cpt = 0
        # Set number of torch threads to 1 to avoid issues with multiprocessing
        original_num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        for _ in range(cfg.nb_eval_eps):
            last_states1: list[OperationState] = []
            last_states2: list[OperationState] = []
            terminated = False
            state = self.reset()
            bench_features = state.bench_features
            while not terminated:
                # Run agent1 on the current state
                process1 = multiprocessing.Process(target=agent1.run_parallel, args=(state, cpt, trajectories_list, 'greedy'))
                processes.append(process1)
                process1.start()
                cpt += 1
                # Run agent2 on the current state
                process2 = multiprocessing.Process(target=agent2.run_parallel, args=(state, cpt, trajectories_list, 'greedy'))
                processes.append(process2)
                process2.start()
                # Take a step in the environment
                state, terminated = self.step(state)
                terminated_list.append(terminated)
                cpt += 1
        for i in tqdm(range(len(processes)), desc="Eval MCTS search"):
            # Wait for trajectory data in queue
            processes[i].join()
            processes[i].close()
        # Reset the number of torch threads
        torch.set_num_threads(original_num_threads)
        # Sort trajectories
        trajectories: list[tuple[int, tuple[OperationState, AASNetworkPolicyEstimation]]] = list(sorted(list(trajectories_list), key=lambda x: x[0]))
        # Get last states
        bench_states1 = []
        bench_states2 = []
        tmp_states1 = []
        tmp_states2 = []
        for i in range(0, len(trajectories), 2):
            trajectory1 = trajectories[i][1]
            trajectory2 = trajectories[i + 1][1]
            terminated = terminated_list[i // 2]
            tmp_states1.append(trajectory1[-1][0])
            tmp_states2.append(trajectory2[-1][0])
            if terminated:
                bench_states1.append(tmp_states1)
                bench_states2.append(tmp_states2)
                tmp_states1 = []
                tmp_states2 = []
        # Execute optimized operations
        for i in tqdm(range(len(bench_states1)), desc="Evaluation execution"):
            last_states1 = bench_states1[i]
            last_states2 = bench_states2[i]
            # Evaluate the transformed code
            exec_time1, assertion1, _ = evaluate_benchmark_code_with_timeout(last_states1, self.tmp_file_path)
            exec_time2, assertion2, _ = evaluate_benchmark_code_with_timeout(last_states2, self.tmp_file_path)
            # Calculate speedups
            root_exec_time = bench_features.root_exec_time
            speedup1 = root_exec_time / exec_time1 if exec_time1 is not None and assertion1 else 1.0
            speedup2 = root_exec_time / exec_time2 if exec_time2 is not None and assertion2 else 1.0
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
            float: The average speedup of the agent on the benchmarks.
        """
        # Run specified number of full benchmark episodes
        speedups = []
        # TODO: make the number of episodes configurable
        for _ in range(1):
            optimized_states = []
            state = self.reset()
            terminated = False
            bench_features = state.bench_features
            while not terminated:
                # Run the agent on the current state
                optimized_state = agent.eval(state, mode=mode)
                # Save the last state in trajectory
                optimized_states.append(optimized_state)
                # Take a step in the environment
                state, terminated = self.step(state)
            # Evaluate the transformed code
            exec_time, assertion, _ = evaluate_benchmark_code_with_timeout(optimized_states, self.tmp_file_path)
            # Calculate speedup
            root_exec_time = bench_features.root_exec_time
            if exec_time is not None and assertion:
                speedup = root_exec_time / exec_time
                print_info(f"Speedup: {speedup}")
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
        state = self.train_env.reset()
        # Loop trough iterations
        for _ in tqdm(range(cfg.nb_iterations), desc="Main Loop"):
            # ======================== Gather training data ========================
            # Get training states
            manager = multiprocessing.Manager()
            trajectories_list = manager.list()
            processes: list[multiprocessing.Process] = []
            print_info("Started MCTS search ...")
            # Set number of torch threads to 1 to avoid issues with multiprocessing
            original_num_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            for j in range(cfg.nb_train_eps):
                # Run the agent on the current state
                process = multiprocessing.Process(target=self.agent.run_parallel, args=(state, j, trajectories_list))
                processes.append(process)
                process.start()
                # Take a step in the environment
                state, _ = self.train_env.step(state)
            for j in tqdm(range(len(processes)), desc="MCTS Search"):
                # Wait for trajectory data in queue
                processes[j].join()
                processes[j].close()
            # Save data gathered per iteration
            trajectories: list[list[tuple[OperationState, AASNetworkPolicyEstimation]]] = [trajectory for _, trajectory in sorted(list(trajectories_list), key=lambda x: x[0])]
            # Reset the number of torch threads
            torch.set_num_threads(original_num_threads)
            print_info("Number of trajectories:", len(trajectories))
            print_info("Total number of data points:", sum(len(trajectory) for trajectory in trajectories))
            print_info("MCTS search ended ...")

            # ======================== Execute trajectories ========================
            print_info("Started execution ...")
            speedups = []
            for j in tqdm(range(len(trajectories)), desc='Trajectories Execution'):
                # Get the last state of the trajectory
                last_state, _ = trajectories[j][-1]
                # Run the code with last state transformation list
                exec_time, assertion, transformed_code = evaluate_code_with_timeout(last_state, self.train_env.tmp_file_path)
                # Get the reward
                value = self.train_env.get_reward(last_state, exec_time)
                # Add the trajectory to the data queue
                if exec_time is not None and assertion:
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
            # Load the previous agent
            self.prev_agent = self.agent.copy()
            # Train the current agent
            print_info("Started training ...")
            train_stats = self.agent.train(list(self.data))
            if neptune_logs is not None:
                neptune_logs['train/selection_loss'].extend(train_stats.selection_loss, wait=True)
                neptune_logs['train/value_loss'].extend(train_stats.value_loss, wait=True)
                for j in range(cfg.max_num_loops):
                    neptune_logs[f'train/parallel_params_loss_{j}'].extend(train_stats.parallel_params_loss[j], wait=True)
            print_info("Training ended ...")

            # ========== Evaluate the agent with MCTS for next training ============
            # Compare between the current agent and the previous one
            print_info("Started comparison ...")
            score, speedups1, speedups2 = self.eval_env.compare(self.agent, self.prev_agent)
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
                print_success("Agent improved, keeping the current agent ...")
                if neptune_logs is not None and len(speedups1) > 0:
                    neptune_logs['eval/comp/final_speedup'].extend(speedups1, wait=True)
            print_info("Comparison ended ...")

            # ================ Evaluate the agent without MCTS =====================
            print_info("Started evaluation ...")
            greedy_speedups = self.eval_env.eval(self.agent, mode='greedy')
            print_info("Greedy speedups average:", sum(greedy_speedups) / len(greedy_speedups) if len(greedy_speedups) > 0 else 0.0)
            stochastic_speedups = self.eval_env.eval(self.agent, mode='stochastic')
            print_info("Stochastic speedups average:", sum(stochastic_speedups) / len(stochastic_speedups) if len(stochastic_speedups) > 0 else 0.0)
            if neptune_logs is not None:
                if len(greedy_speedups) > 0:
                    neptune_logs['eval/greedy/final_speedup'].extend(greedy_speedups, wait=True)
                if len(stochastic_speedups) > 0:
                    neptune_logs['eval/stochastic/final_speedup'].extend(stochastic_speedups, wait=True)
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
