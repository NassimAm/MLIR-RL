from aas import config as cfg
from aas.observation import BenchmarkFeatures, extract_bench_features_from_file, extract_bench_features_from_code
from aas.state import OperationState
from aas.agent import AlphaAutoScheduler
from typing import Optional
import random
import string
import os
import json
from tqdm import tqdm
from utils.log import print_error, print_success


class AASTrainer:
    """Environment for training an AlphaAutoScheduler agent."""

    benchmarks_data: list[tuple[str, BenchmarkFeatures]]
    """Lists for each benchmark the benchmark's name and its features."""
    truncate: int
    """The maximum number of steps in the schedule."""
    reset_repeat: int
    """The number of times to repeat the reset function."""
    step_repeat: int
    """The number of times to repeat the step function."""
    tmp_file: str
    """The temporary file to store the intermediate representations."""

    def __init__(self, reset_repeat: int = 1, step_repeat: int = 1, tmp_file_path: Optional[str] = None):
        """Initialize the environment.

        Args:
            reset_repeat (int): The number of times to repeat the reset function. Defaults to 1.
            step_repeat (int): The number of times to repeat the step function. Defaults to 1.
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
                benchmark_data = extract_bench_features_from_file(bench_name, bench_file, exec_time, exec_time)
                self.benchmarks_data.append((bench_name, benchmark_data))
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
                benchmark_data = extract_bench_features_from_code(bench_name, code, exec_time, exec_time)
                self.benchmarks_data.append((bench_name, benchmark_data))

        self.reset_repeat = reset_repeat
        self.step_repeat = step_repeat

    def reset(self, idx: Optional[int] = None):
        """Reset the environment.

        Args:
            idx (Optional[int]): The index of the benchmark to set the environement to. If None, a random benchmark is selected. Defaults to None.

        Returns:
            OperationState: The initial state of the environment.
        """
        if idx is not None:
            # We get the operation with the right index
            self.bench_index = idx
        else:
            # Get a random operation
            self.bench_index = random.randint(0, len(self.benchmarks_data) - 1)
        bench_name, benchmark_data = self.benchmarks_data[self.bench_index]

        # The number of loops in the Linalg operations
        if cfg.data_format == "mlir":
            # Get benchmark file
            bench_file = os.path.join(cfg.benchmarks_folder_path, bench_name + ".mlir")
            # Reload original code and features
            benchmark_data = extract_bench_features_from_file(bench_name, bench_file, benchmark_data.root_exec_time, benchmark_data.root_exec_time)
            self.benchmarks_data[self.bench_index] = (bench_name, benchmark_data)
        # TODO: Add case where data_format is "json" and reload data from json file if needed (if optimization mode is "all")

        # Get the last operation
        operation_tag = benchmark_data.operation_tags[-1]
        operation_features = benchmark_data.operations[operation_tag]

        state = OperationState(
            bench_name=bench_name,
            operation_tag=operation_tag,
            operation_features=operation_features,
            transformed_code=benchmark_data.code,
            step_count=0,
            transformation_history=[]
        )

        return state

    def step(self, agent: AlphaAutoScheduler, state: OperationState):
        """Take a step in the environment given an agent.

        Args:
            agent (AlphaAutoScheduler): The agent to use to take the step.
            state (OperationState): The current state of the environment.

        Returns:
            OperationState: The next state of the environment.
            bool: Whether the benchmark optimization is over or not.
            float: The speedup obtained by the agent for the benchmark.
        """

        # Get benchmark data
        bench_name, bench_data = self.benchmarks_data[self.bench_index]

        # Run the agent
        optimized_state, optimized_exec_time, assertion = agent.run(state, bench_data.exec_time)
        # Print infos and update reward
        if optimized_exec_time is None:
            print_error(f"EXECUTION ERROR ({optimized_state.bench_name} {optimized_state.operation_tag}): {optimized_state.transformation_history}")
        else:
            if assertion:
                print_success(f"RELATIVE SPEEDUP ({optimized_state.bench_name} {optimized_state.operation_tag}): {bench_data.exec_time / optimized_exec_time}")
                print_success(f"ABSOLUTE SPEEDUP: {bench_data.root_exec_time / optimized_exec_time}")
                print_success(f"OLD EXECUTION TIME: {bench_data.exec_time}")
                print_success(f"NEW EXECUTION TIME: {optimized_exec_time}")
                print_success("ACTIONS", optimized_state.transformation_history)
            else:
                print_error(f"ASSERTION FAILED ({optimized_state.bench_name} {optimized_state.operation_tag}): {optimized_state.transformation_history}")

        # Indicates that the benchmark optimization is over or not
        terminated = True
        if cfg.optimization_mode == "all":
            op_index = bench_data.operation_tags.index(optimized_state.operation_tag)
            if op_index > 0:
                # Benchmark optimization is not over
                terminated = False
                # Re-extract operations data from the new code
                new_bench_data = extract_bench_features_from_code(bench_name, optimized_state.transformed_code, bench_data.root_exec_time, optimized_exec_time)
                self.benchmarks_data[self.bench_index] = (bench_name, new_bench_data)
                # Build a new state that points to the next operation
                new_op_tag = new_bench_data.operation_tags[op_index - 1]
                new_op_features = new_bench_data.operations[new_op_tag]
                next_state = OperationState(
                    bench_name=bench_name,
                    operation_tag=new_op_tag,
                    operation_features=new_op_features,
                    transformed_code=new_bench_data.code,
                    step_count=0,
                    transformation_history=[]
                )

        # If the benchmark optimization is over, we reset the environment and get the full speedup
        speedup = 1.0
        if terminated:
            if optimized_exec_time is not None:
                speedup = bench_data.root_exec_time / optimized_exec_time
            else:
                speedup = 1.0
            next_state = self.reset()

        return next_state, terminated, speedup
