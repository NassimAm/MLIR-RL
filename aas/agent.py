from aas import config as cfg
from aas.nn import AASNetwork
from aas.wrappers import AASNetworkWrapper, AASNetworkPolicyEstimation, AASNetworkEstimation
from aas.node import Node
from aas.mcts import MCTS
from aas.state import OperationState
from typing import Optional, Callable, Literal, Iterable
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
            # max_q_child = max([child for child in node.children], key=lambda x: x.q, default=None)
            # max_nb_visits_child = max([child for child in node.children], key=lambda x: x.nb_visits, default=None)
            # print(node.state.operation_tag, "Max Q", max_q_child.to_str(mcts.min_value, mcts.max_value, mcts.c_puct) if max_q_child is not None else None)
            # print(node.state.operation_tag, "Fisrt child", node.children[0].to_str(mcts.min_value, mcts.max_value, mcts.c_puct) if len(node.children) > 0 else None)
            # print(node.state.operation_tag, "Max nb visits", max_nb_visits_child.to_str(mcts.min_value, mcts.max_value, mcts.c_puct))
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

    def run_parallel(self, args: Iterable):
        """Run the Alpha AutoScheduler on a given state and return training data about the trajectory. This ùethod should be used
        instead of the original one when doing multiprocessing.

        Args:
            args (Iterable): The arguments to pass to the run method.

        Returns:
            list[tuple[OperationState, AASNetworkPolicyEstimation]]: The trajectory taken by the agent.
        """
        # Reseed numpy random generator
        np.random.seed()
        # Reseed torch random generator
        torch.manual_seed(np.random.randint(0, 2**32 - 1))
        # Run agent
        return self.run(*args)

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
        print(state.operation_tag)
        while not state.is_terminal():
            # Get the estimation of the current state
            aas_estimation = self.network_wrapper.eval(state)
            print(aas_estimation)
            # Get the action to take
            action = aas_estimation.policy.get_action_from_hierarchical_probs(mode=mode)
            print(action)
            # Apply the action to the state
            state = state.next(action)
            print("Action Value", self.network_wrapper.eval(state).get_value())
        print("=====================================")
        # Return the final state
        return state

    def save(self, path: str):
        """Save the Alpha AutoScheduler to a file.

        Args:
            path (str): The path to save the network to.
        """
        torch.save(self.network.state_dict(), path)

    def load_from_file(path: str, reward_func: Callable[[OperationState, int], float]):
        """Load the Alpha AutoScheduler from a file.

        Args:
            path (str): The path to load the network from.

        Returns:
            AlphaAutoScheduler: The loaded Alpha AutoScheduler.
        """
        network = AASNetwork()
        network.load_state_dict(torch.load(path, weights_only=True))
        return AlphaAutoScheduler(reward_func, network=network)

    def copy(self):
        """Copy the Alpha AutoScheduler.

        Returns:
            AlphaAutoScheduler: The copied Alpha AutoScheduler.
        """
        return deepcopy(self)
