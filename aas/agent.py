from aas import config as cfg
from aas.nn import AASNetwork
from aas.wrappers import AASNetworkManager, AASNetworkEstimation
from aas.node import Node
from aas.mcts import MCTS
from aas.state import OperationState
from aas.evaluation import evaluate_code_with_timeout
import math
import torch
from typing import Optional


class AlphaAutoSchedulerStats:
    """The Alpha AutoScheduler Stats class. It contains all collected stats during training if logging is enabled."""
    ...


class AlphaAutoScheduler:
    """The Alpha AutoScheduler class."""

    def __init__(self, tmp_file_path: str):
        """Initialize the Alpha AutoScheduler.

        Args:
            tmp_file_path (str): The temporary file path to write the MLIR code to.
        """
        self.network = AASNetwork()
        self.network_manager = AASNetworkManager(self.network)
        self.mcts = MCTS(self.network_manager, tmp_file_path)
        self.tmp_file_path = tmp_file_path
        self.stats = AlphaAutoSchedulerStats()

    def run(self, state: OperationState, root_exec_time: int) -> tuple[OperationState, Optional[int], bool]:
        """Run the Alpha AutoScheduler on a given state.

        Args:
            state (OperationState): The state to run the Alpha AutoScheduler on.
            root_exec_time (int): Execution time of the code right before optimizing the operation represented by the state given.

        Returns:
            OperationState: The state after running the Alpha AutoScheduler.
            Optional[int]: The execution time of the optimized code.
            bool: Whether the assertion was successful.
        """
        # Create an MCTS tree with the given state
        root = Node(state)
        node = root
        # Save the trajectory taken by MCTS
        trajectory: list[tuple[OperationState, AASNetworkEstimation]] = []
        # Run MCTS searches until a terminal node is reached
        while not node.is_terminal():
            # Get MCTS policy target
            target_policy_estimation, next_node = self.mcts.run(node, n_iterations=cfg.mcts_nb_iterations)
            # Save the current state and the target policy estimation and set value to 0 for now
            trajectory.append((node.state, AASNetworkEstimation(
                policy=target_policy_estimation,
                value=torch.tensor(0.0)
            )))
            # Make the next node the root node
            next_node.node_p = 1.0
            next_node.parent = None
            node = next_node
        # Evaluate the code
        # TODO: Assertion should always be true (do something to check this)
        exec_time, assertion = evaluate_code_with_timeout(node.state, self.tmp_file_path)
        # If the code execution was successful and the assertion is true
        if (exec_time is not None) and assertion:
            # Get target value
            target_value = self.get_speedup_reward(root_exec_time, exec_time)
            # Update trajectory with target value
            for _, aas_estimation in trajectory:
                aas_estimation.value = torch.tensor(target_value)
            # Train the model on the trajectory
            self.network_manager.train_on_trajectory(trajectory)

        return node.state, exec_time, assertion

    def get_speedup_reward(self, root_exec_time: int, exec_time: int):
        """Get the speedup reward based on the execution time.

        Args:
            root_exec_time (int): The execution time of the root node.
            exec_time (int): The execution time of the current node.

        Returns:
            float: The speedup reward.
        """
        return math.log(root_exec_time / exec_time, 10)
