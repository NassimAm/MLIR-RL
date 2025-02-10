from aas import config as cfg
from aas.nn import AASNetwork
from aas.wrappers import AASNetworkWrapper, AASNetworkPolicyEstimation, AASNetworkEstimation
from aas.node import Node
from aas.mcts import MCTS
from aas.state import OperationState
from typing import Optional, Callable, Literal
import multiprocessing.managers
from copy import deepcopy
import numpy as np
import torch


class AlphaAutoSchedulerStats:
    """The Alpha AutoScheduler Stats class. It contains all collected stats during training if logging is enabled."""
    ...


class AlphaAutoScheduler:
    """The Alpha AutoScheduler class."""

    def __init__(self, reward_func: Callable[[OperationState, int], float], network: Optional[AASNetwork] = None):
        """Initialize the Alpha AutoScheduler.

        Args:
            reward_func (Callable[[OperationState, int], float]): The reward function used by the training environment
            network (Optional[AASNetwork], optional): The network to use. Defaults to None.
        """
        self.reward_func = reward_func
        if network is None:
            self.network = AASNetwork()
        else:
            self.network = network
        self.network_wrapper = AASNetworkWrapper(self.network)
        # self.stats = AlphaAutoSchedulerStats()

    def run(self, state: OperationState, mode: Literal['greedy', 'stochastic'] = 'stochastic'):
        """Run the Alpha AutoScheduler on a given state and return training data about the trajectory taken by the agent.

        Args:
            state (OperationState): The initial operation state to optimize.
            mode (Literal['greedy', 'stochastic'], optional): The mode to run the agent. Defaults to 'stochastic'.

        Returns:
            list[tuple[OperationState, AASNetworkPolicyEstimation]]: The trajectory taken by the agent.
        """
        # Create an MCTS tree with the given state
        root = Node(state)
        node = root
        # Save the trajectory taken by MCTS
        trajectory: list[tuple[OperationState, AASNetworkPolicyEstimation]] = []
        # Reset the MCTS algorithm
        mcts = MCTS(self.network_wrapper, self.reward_func)
        # Run MCTS searches until a terminal node is reached
        while not node.is_terminal():
            # Get MCTS policy target
            target_policy_estimation, next_node = mcts.run(node, n_iterations=cfg.mcts_nb_iterations, mode=mode)
            # Save the current state and the target policy estimation and set value to 0 for now
            trajectory.append((node.state, target_policy_estimation))
            # Make the next node the root node
            next_node.node_exploration_factor = 1.0
            next_node.parent = None
            node = next_node
        # Add the terminal node to the trajectory
        trajectory.append((node.state, self.network_wrapper.get_no_action_aas_policy_estimation()))
        # Return the trajectory
        return trajectory

    def run_parallel(self, state: OperationState, process_id: int, trajectories_list: multiprocessing.managers.ListProxy, mode: Literal['greedy', 'stochastic'] = 'stochastic'):
        """Run the Alpha AutoScheduler on a given state and return training data about the trajectory taken by the agent
        and put in a multiprocessing queue.

        Args:
            state (OperationState): The initial operation state to optimize.
            process_id (int): The process id of the current process.
            trajectories_list (multiprocessing.managers.ListProxy): The list to put the trajectory in.
            mode (Literal['greedy', 'stochastic'], optional): The mode to run the agent. Defaults to 'stochastic'.
        """
        # Reseed numpy random generator
        np.random.seed()
        # Reseed torch random generator
        torch.manual_seed(np.random.randint(0, 2**32 - 1))
        # Run agent
        trajectory = self.run(deepcopy(state), mode=mode)
        trajectories_list.append((process_id, trajectory))

    def train(self, data: list[tuple[OperationState, AASNetworkEstimation]]):
        """Train the Alpha AutoScheduler on given history data.

        Args:
            data (list[tuple[OperationState, AASNetworkEstimation]]): The history data to train the agent.
        """
        self.network_wrapper.train(data)
        return self.network_wrapper.stats

    def eval(self, state: OperationState, mode: Literal['greedy', 'stochastic'] = 'stochastic'):
        """Evaluate the Alpha AutoScheduler on a given state.

        Args:
            state (OperationState): The initial operation state to evaluate.
            mode (Literal['greedy', 'stochastic'], optional): The mode to evaluate the agent. Defaults to 'stochastic'.

        Returns:
            OperationState: The final state after the agent has evaluated the state and taken actions.
        """
        while not state.is_terminal():
            # Get the estimation of the current state
            aas_estimation = self.network_wrapper.eval(state)
            # Get the action to take
            action = aas_estimation.policy.get_max_hierarchical_prob_action(mode=mode)
            # Apply the action to the state
            state = state.next(action)
        # Return the final state
        return state

    def save(self, path: str):
        """Save the Alpha AutoScheduler to a file.

        Args:
            path (str): The path to save the network to.
        """
        self.network.save(path)

    def load_from_file(path: str, reward_func: Callable[[OperationState, int], float]):
        """Load the Alpha AutoScheduler from a file.

        Args:
            path (str): The path to load the network from.

        Returns:
            AlphaAutoScheduler: The loaded Alpha AutoScheduler.
        """
        network = AASNetwork()
        network.load(path)
        return AlphaAutoScheduler(reward_func, network=network)
