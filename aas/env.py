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
from utils.log import print_info, print_alert, print_success


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

        # # Run the agent
        # optimized_state, reward, optimized_exec_time, assertion, optimized_code = agent.run(bench_data, state)
        # # Print infos and update reward
        # if optimized_exec_time is None:
        #     if optimized_code:
        #         print_error(f"EXECUTION ERROR: ({optimized_state.bench_name} {optimized_state.operation_tag})")
        #         print_error("ACTIONS:", optimized_state.transformation_history)
        #     else:
        #         print_error(f"TRANSFORMATION ERROR: ({optimized_state.bench_name} {optimized_state.operation_tag})")
        #         print_error("ACTIONS:", optimized_state.transformation_history)
        # else:
        #     if assertion:
        #         # Get relative speedup
        #         relative_speedup = bench_data.exec_time / optimized_exec_time
        #         # Update best speedup for the current operation
        #         if relative_speedup > bench_data.best_speedup[optimized_state.operation_tag]:
        #             bench_data.best_speedup[optimized_state.operation_tag] = relative_speedup
        #         # Print logs
        #         print_success(f"SUCCESS: ({optimized_state.bench_name} {optimized_state.operation_tag})")
        #         print_success("RELATIVE SPEEDUP:", relative_speedup)
        #         print_success("ABSOLUTE SPEEDUP:", bench_data.root_exec_time / optimized_exec_time)
        #         print_success("OLD EXECUTION TIME:", bench_data.exec_time)
        #         print_success("NEW EXECUTION TIME:", optimized_exec_time)
        #         print_success("BEST SPEEDUP:", bench_data.best_speedup[optimized_state.operation_tag])
        #         print_success("REWARD:", reward)
        #         print_success("ACTIONS:", optimized_state.transformation_history)
        #     else:
        #         print_error(f"ASSERTION FAILED: ({optimized_state.bench_name} {optimized_state.operation_tag})")
        #         print_error("ACTIONS:", optimized_state.transformation_history)

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
        """
        # Run specified number of full benchmark episodes
        score = 0
        state = self.reset()
        terminated = False
        for _ in range(cfg.nb_eval_eps):
            last_states1 = []
            last_states2 = []
            bench_features = state.bench_features
            while not terminated:
                # Run agent1 on the current state
                trajectory = agent1.run(state, mode='greedy')
                last_state, _ = trajectory[-1]
                # Save the last state in trajectory
                last_states1.append(last_state)
                # Run agent2 on the current state
                trajectory = agent2.run(state, mode='greedy')
                last_state, _ = trajectory[-1]
                # Save the last state in trajectory
                last_states2.append(last_state)
                # Take a step in the environment
                state, terminated = self.step(state)
            # Evaluate the transformed code
            exec_time1, assertion1, _ = evaluate_benchmark_code_with_timeout(last_states1, self.tmp_file_path)
            exec_time2, assertion2, _ = evaluate_benchmark_code_with_timeout(last_states2, self.tmp_file_path)
            # Calculate speedups
            root_exec_time = bench_features.root_exec_time
            speedup1 = root_exec_time / exec_time1 if exec_time1 is not None and assertion1 else 1.0
            speedup2 = root_exec_time / exec_time2 if exec_time2 is not None and assertion2 else 1.0
            # Calculate score based on ratio between speedups (positive score = keep new agent, negative score = keep old agent)
            score += speedup1 / speedup2 if speedup1 >= speedup2 else -speedup2 / speedup1
        # Return the score
        return score


class AASTrainer:
    """Trainer for an AlphaAutoScheduler agent."""

    env: AASOpEnv
    """The environment to train the agent on."""
    agent: AlphaAutoScheduler
    """The agent agent to train."""
    prev_agent: Optional[AlphaAutoScheduler]
    """The previous agent to compare with the current one."""

    def __init__(self, env_type: Literal["op"] = "op"):
        """Initialize the trainer.

        Args:
            env (AASOpEnv): The environment to train the agent on.
        """
        # Initialize the environment
        if env_type == "op":
            self.env = AASOpEnv()
        else:
            raise ValueError(f"Invalid environment type ({env_type}). Please choose 'op' for operation-wise optimization.")
        # Initialize agents
        self.save_file_path = os.path.join("models", "aas_agent.pt")
        self.agent = AlphaAutoScheduler(self.env.get_reward)
        self.agent.save(self.save_file_path)
        self.prev_agent = None
        # Initialize data queue
        self.data = deque(maxlen=cfg.data_queue_max_length)

    def train(self):
        """Train the agent to optimize benchmarks."""
        # Initialize the environment
        state = self.env.reset()
        # Loop trough iterations
        for _ in tqdm(range(cfg.nb_iterations), desc="Main Loop"):
            # Save data gathered per iteration
            trajectories: list[list[tuple[OperationState, AASNetworkPolicyEstimation]]] = []
            # Run train episodes
            print_info("Started MCTS search ...")
            for _ in tqdm(range(cfg.nb_train_eps), desc="MCTS Search"):
                # Run the agent on the current state
                trajectory = self.agent.run(state)
                # Save the trajectory
                trajectories.append(trajectory)
                # Take a step in the environment
                state, _ = self.env.step(state)
            print_info("Number of trajectories:", len(trajectories))
            print_info("Total number of data points:", sum(len(trajectory) for trajectory in trajectories))
            print_info("MCTS search ended ...")
            # Evaluate trajectories
            print_info("Started execution ...")
            speedups = []
            for j in tqdm(range(len(trajectories)), desc="Trajectory execution"):
                # Get the last state of the trajectory
                last_state, _ = trajectories[j][-1]
                # Run the code with last state transformation list
                exec_time, assertion, _ = evaluate_code_with_timeout(last_state, self.env.tmp_file_path)
                # Get the reward
                value = self.env.get_reward(last_state, exec_time)
                # Add the trajectory to the data queue
                if exec_time is not None and assertion:
                    speedups.append(last_state.bench_features.root_exec_time / exec_time)
                    for trajectory_state, trajectory_policy in trajectories[j]:
                        self.data.append((trajectory_state, AASNetworkEstimation(
                            policy=trajectory_policy,
                            value=value
                        )))
            print_info("Average speedup:", sum(speedups) / len(speedups))
            print_info("Max speedup:", max(speedups))
            print_info("Execution ended ...")
            # Load the previous agent
            self.prev_agent = AlphaAutoScheduler.load_from_file(self.save_file_path, self.env.get_reward)
            # Train the current agent
            print_info("Started training ...")
            self.agent.train(list(self.data))
            print_info("Training ended ...")
            # Compare between the current agent and the previous one
            print_info("Started comparison ...")
            score = self.env.compare(self.agent, self.prev_agent)
            # If the agent did not improve, load the previous agent
            if score < 0:
                # Load the previous agent
                self.agent = self.prev_agent
                print_alert("Agent did not improve, loading the previous agent ...")
            else:
                print_success("Agent improved, keeping the current agent ...")
            print_info("Comparison ended ...")
            # Save the best agent so far
            self.agent.save(self.save_file_path)

    def get_best_agent(self):
        """Load the best agent so far.

        Returns:
            AlphaAutoScheduler: The best agent so far.
        """
        return AlphaAutoScheduler.load_from_file(self.save_file_path, self.env.get_reward)
