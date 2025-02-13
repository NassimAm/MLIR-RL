from aas import config as cfg
from aas.wrappers import AASNetworkWrapper
from aas.node import Node
from aas.state import OperationState
from aas.evaluation import get_cached_exec_time
import numpy as np
from typing import Callable, Literal


class MCTS:
    """Class to represent the Monte Carlo Tree Search algorithm"""

    aas_network_wrapper: AASNetworkWrapper
    """The AlphaAutoScheduler network wrapper to calculate the prior probabilities of the actions and node values."""
    c_puct: float
    """The PUCT constant for the MCTS algorithm"""
    action_temperature: float
    """The temperature for the MCTS action probabilities"""
    action_temperature_decay: float
    """The temperature decay for the MCTS action probabilities"""

    def __init__(self, aas_network_wrapper: AASNetworkWrapper, reward_func: Callable[[OperationState, int], float]):
        """Initialize the MCTS algorithm.

        Args:
            aas_network_wrapper (AASNetworkWrapper): The AlphaAutoScheduler network wrapper to calculate the prior probabilities of the actions and node values.
            reward_func (Callable[[OperationState, int], float]): The reward function used by the training environment.
        """
        self.aas_network_wrapper = aas_network_wrapper
        self.reward_func = reward_func
        self.c_puct = cfg.mcts_c_puct
        self.action_temperature = 1.0
        self.action_temperature_decay = cfg.mcts_action_temperature_decay

    def reset(self):
        """Reset the MCTS algorithm."""
        self.action_temperature = 1.0

    def select(self, root: Node) -> Node:
        """Select a node in the MCTS tree to explore.

        Args:
            root (Node): The root node of the MCTS tree.

        Returns:
            Node: The selected node.
        """
        node = root
        while len(node.children) > 0:
            # Get dirichlet noise for the children
            noises = np.random.dirichlet([0.03] * len(node.children))
            # Calculate S scores
            scores = np.array([child.get_s_score(self.c_puct, noise=noises[i].item()) for i, child in enumerate(node.children)])
            # Get the child node with the highest S score with random tie breaking
            node_id = np.random.choice(np.flatnonzero(scores == scores.max())).item()
            node = node.children[node_id]
        return node

    def expand(self, node: Node):
        """Expand a node in the MCTS tree by adding its children and evaluate it with value approximation.

        Args:
            node (Node): The node to expand.
        """
        # Evaluate the node policy
        aas_policy_estimation = self.aas_network_wrapper.eval_node_policy(node)
        # Get available actions
        available_actions = node.get_available_actions()
        sum_child_node_factors = 0.0
        for action in available_actions:
            # Get next state
            next_state = node.state.next(action)
            # Process child node exploration factor
            child_node_factor = self.aas_network_wrapper.get_action_prob(node.state, action, aas_policy_estimation)
            sum_child_node_factors += child_node_factor
            # Add child node to the tree
            node.add_child(next_state, child_node_factor)
        # Normalize node factors and get children values
        for child in node.children:
            # Normalize the child node exploration factor
            child.node_exploration_factor /= sum_child_node_factors
            # Get the child node value
            real_exec_time = get_cached_exec_time(child.state)
            if real_exec_time is not None:
                # Use the real speedup if available to get the value
                node_value = self.reward_func(child.state, real_exec_time)
            else:
                # Otherwise, use the AAS network to estimate the value
                node_value = self.aas_network_wrapper.eval_node_value(child)
            # Update the node with its value
            child.update_leaf(node_value)

    def backpropagate(self, node: Node):
        """Backpropagate the speedup value up the MCTS tree.

        Args:
            node (Node): The node to backpropagate from.
        """
        # Increment the number of visits of the node
        node.update_visits_count()
        # Backpropagate the value up the tree
        parent = node.parent
        child = node
        while parent is not None:
            parent.update(child.q)
            child = parent
            parent = parent.parent

    def run(self, root: Node, n_iterations: int, mode: Literal['greedy', 'stochastic'] = 'stochastic'):
        """Perform the MCTS search for a given number of iterations and convert
         action probabilities to AASNetwork policy estimation.

        Args:
            root (Node): The root node of the MCTS tree.
            n_iterations (int): The number of iterations to perform the search.
            mode (Literal['greedy', 'stochastic'], optional): The mode to use for the action probabilities. Defaults to 'stochastic'.

        Returns:
            AASNetworkPolicyEstimation: The AASNetwork policy estimation.
            Node: The child node with the highest MCTS probability. The root is returned if no children.
        """
        # Run the MCTS search for a given number of iterations
        for _ in range(n_iterations):
            node = self.select(root)
            self.expand(node)
            self.backpropagate(node)
        # Calculate next policy estimation
        aas_policy_estimation, max_p_node = self.aas_network_wrapper.evaluate_tree(root, self.action_temperature, mode=mode)
        # Update temperatures
        self.action_temperature *= self.action_temperature_decay
        # Return the full AAS policy estimation
        return aas_policy_estimation, max_p_node
