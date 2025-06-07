from aas import config as cfg
from aas.wrappers import AASNetworkWrapper
from aas.node import Node
from aas.state import OperationState
from aas.evaluation import get_cached_exec_time
import numpy as np
from typing import Callable, Literal, Optional
import math


class MCTS:
    """Class to represent the Monte Carlo Tree Search algorithm"""

    aas_network_wrapper: AASNetworkWrapper
    """The AlphaAutoScheduler network wrapper to calculate the prior probabilities of the actions and node values."""
    c_puct: float
    """The PUCT constant for the MCTS algorithm"""
    action_temperature: float
    """The temperature for the MCTS action probabilities"""

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
        self.max_value = -math.inf
        self.min_value = math.inf

    def reset(self):
        """Reset the MCTS algorithm."""
        self.action_temperature = 1.0

    def set_temperature(self, temperature: float):
        """Set the temperature for the MCTS action probabilities.

        Args:
            temperature (float): The temperature for the MCTS action probabilities.
        """
        self.action_temperature = temperature

    def select(self, root: Node) -> Node:
        """Select a node in the MCTS tree to explore.

        Args:
            root (Node): The root node of the MCTS tree.

        Returns:
            Node: The selected node.
        """
        node = root
        while len(node.children) > 0:
            # Get dirichlet noise for root children
            noises = None
            if node.parent is None:
                noises = np.random.dirichlet([0.03] * len(node.children))
            # Calculate S scores
            scores = np.array([child.get_s_score(self.min_value, self.max_value, self.c_puct, noise=None if noises is None else noises[i].item()) for i, child in enumerate(node.children)])
            # Get the child node with the highest S score with random tie breaking
            node_id = np.random.choice(np.flatnonzero(np.isclose(scores, scores.max()))).item()
            node = node.children[node_id]
        return node

    def expand(self, node: Node, exec_db: Optional[dict] = None):
        """Expand a node in the MCTS tree by adding its children and evaluate it with value approximation.

        Args:
            node (Node): The node to expand.
            exec_db (Optional[dict], optional): The benchmark execution database to use for the speedup values. Defaults to None.
        """
        # Evaluate the node
        aas_estimation = self.aas_network_wrapper.eval_node(node)
        aas_estimation_policy = aas_estimation.policy
        # Get the node value
        real_exec_time = get_cached_exec_time(exec_db, node.state)
        if real_exec_time is not None:
            # Use the real speedup if available to get the value
            node_value = self.reward_func(node.state, real_exec_time)
        else:
            # Otherwise, use the AAS network estimation to get the value
            node_value = aas_estimation.get_value().item()
        # Update the node with its value
        node.update_leaf(node_value)
        self.min_value = min(self.min_value, node_value)
        self.max_value = max(self.max_value, node_value)
        # Get available actions
        available_actions = node.get_available_actions()
        sum_child_node_factors = 0.0
        for action in available_actions:
            # Get next state
            next_state = node.state.next(action)
            # Process child node exploration factor
            child_node_factor = self.aas_network_wrapper.get_action_prob(node.state, action, aas_estimation_policy)
            sum_child_node_factors += child_node_factor
            # Add child node to the tree
            node.add_child(next_state, child_node_factor)
        # Normalize node factors and get children values
        for child in node.children:
            # Normalize the child node exploration factor
            child.node_exploration_factor /= sum_child_node_factors

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

    def run(self, root: Node, n_iterations: int, exec_db: Optional[dict] = None, mode: Literal['greedy', 'stochastic'] = 'stochastic'):
        """Perform the MCTS search for a given number of iterations and convert
         action probabilities to AASNetwork policy estimation.

        Args:
            root (Node): The root node of the MCTS tree.
            n_iterations (int): The number of iterations to perform the search.
            exec_db (Optional[dict], optional): The benchmark execution database to use for the speedup values. Defaults to None.
            mode (Literal['greedy', 'stochastic'], optional): The mode to use for the action probabilities. Defaults to 'stochastic'.

        Returns:
            AASNetworkPolicyEstimation: The AASNetwork policy estimation.
            Node: The child node with the highest MCTS probability. The root is returned if no children.
        """
        # Update min and max values
        self.min_value = min(self.min_value, root.q)
        self.max_value = max(self.max_value, root.q)
        # Run the MCTS search for a given number of iterations
        for _ in range(n_iterations):
            node = self.select(root)
            self.expand(node, exec_db)
            self.backpropagate(node)
        # Calculate next policy estimation
        aas_policy_estimation, max_p_node = self.aas_network_wrapper.evaluate_tree(root, self.action_temperature, mode=mode)
        # Return the full AAS policy estimation
        return aas_policy_estimation, max_p_node
