from aas import config as cfg
from aas.wrappers import AASNetworkManager
from aas.node import Node
from aas.observation.benchmark import BenchmarkFeatures
from aas.evaluation import get_cached_exec_time
import numpy as np


class MCTS:
    """Class to represent the Monte Carlo Tree Search algorithm"""

    aas_network_manager: AASNetworkManager
    """The AlphaAutoScheduler network manager to calculate the prior probabilities of the actions and node values."""
    c_puct: float
    """The PUCT constant for the MCTS algorithm"""
    action_temperature: float
    """The temperature for the MCTS action probabilities"""
    action_temperature_decay: float
    """The temperature decay for the MCTS action probabilities"""
    random_exploration_temperature: float
    """The temperature for the MCTS random exploration"""
    random_exploration_temperature_decay: float
    """The temperature decay for the MCTS random exploration"""

    def __init__(self, aas_network_manager: AASNetworkManager, tmp_file_path: str):
        """Initialize the MCTS algorithm.

        Args:
            aas_network_manager (AASNetworkManager): The AlphaAutoScheduler network manager to calculate the prior probabilities of the actions and node values.
            tmp_file_path (str): The temporary file path to write the MLIR code to.
        """
        self.aas_network_manager = aas_network_manager
        self.tmp_file_path = tmp_file_path
        self.c_puct = cfg.mcts_c_puct
        self.action_temperature = 1.0
        self.action_temperature_decay = cfg.mcts_action_temperature_decay
        self.random_exploration_temperature = 1.0
        self.random_exploration_temperature_decay = cfg.mcts_random_exploration_temperature_decay

    def select(self, root: Node) -> Node:
        """Select a node in the MCTS tree to explore.

        Args:
            root (Node): The root node of the MCTS tree.

        Returns:
            Node: The selected node.
        """
        node = root
        while node.children:
            # Get the child node with the highest S score with random tie breaking
            scores = np.array([child.get_s_score(self.c_puct) for child in node.children])
            node_id = np.random.choice(np.flatnonzero(scores == scores.max())).item()
            node = node.children[node_id]
        return node

    def expand(self, bench_features: BenchmarkFeatures, node: Node):
        """Expand a node in the MCTS tree by adding its children and evaluate it with value approximation.

        Args:
            bench_features (BenchmarkFeatures): The benchmark features.
            node (Node): The node to expand.
        """
        # Evaluate the node
        # TODO: It would be useful if v is always an underestimation of the real speedup
        real_exec_time = get_cached_exec_time(bench_features, node.state)
        if real_exec_time is not None:
            # Use the real speedup if available to get the value
            node_value = self.aas_network_manager.get_speedup_reward(bench_features, real_exec_time)
        else:
            # Otherwise, use the AAS network to estimate the value
            aas_estimation = self.aas_network_manager.eval_node(node)
            node_value = aas_estimation.get_value()
        # Update the node with its value
        node.update_leaf(node_value)
        # Get available actions
        available_actions = node.get_available_actions()
        child_node_factors = []
        for action in available_actions:
            # Get next state
            next_state = node.state.next(action)
            # Process child node exploration factor
            child_node_factor = self.aas_network_manager.get_action_prob(node.state, action, aas_estimation)
            child_node_factors.append(child_node_factor)
            # child_node_factor = self.random_exploration_temperature * 1 + (1 - self.random_exploration_temperature) * child_node_p
            # Add child node to the tree
            node.add_child(next_state, child_node_factor)
        # Normalize node factors and add dirichlet noise
        child_node_factors = np.array(child_node_factors)
        child_node_factors = child_node_factors / np.sum(child_node_factors)
        child_node_factors = 0.75 * child_node_factors + 0.25 * np.random.dirichlet([0.03] * len(available_actions))
        for i, child in enumerate(node.children):
            child.node_exploration_factor = child_node_factors[i].item()

    def backpropagate(self, node: Node):
        """Backpropagate the speedup value up the MCTS tree.

        Args:
            node (Node): The node to backpropagate from.
        """
        while node is not None:
            node.update(node.q)
            node = node.parent

    def run(self, bench_features: BenchmarkFeatures, root: Node, n_iterations: int):
        """Perform the MCTS search for a given number of iterations and convert
         action probabilities to AASNetwork policy estimation.

        Args:
            bench_features (BenchmarkFeatures): The benchmark features.
            root (Node): The root node of the MCTS tree.
            n_iterations (int): The number of iterations to perform the search.

        Returns:
            AASNetworkPolicyEstimation: The AASNetwork policy estimation.
            Node: The child node with the highest MCTS probability. The root is returned if no children.
        """
        # Run the MCTS search for a given number of iterations
        for i in range(n_iterations):
            # print("MCTS iteration", i, '-' * 20)
            node = self.select(root)
            self.expand(bench_features, node)
            self.backpropagate(node)
        print("Root node:", root.to_str(self.c_puct))
        # Calculate next policy estimation
        aas_policy_estimation, max_p_node = self.aas_network_manager.evaluate_tree(root, self.action_temperature)
        print("Selected node:", max_p_node.to_str(self.c_puct))
        print("Max nb visits node:", max(root.children, key=lambda x: x.nb_visits).to_str(self.c_puct))
        print("Max q node:", max(root.children, key=lambda x: x.q).to_str(self.c_puct))
        # Update temperatures
        self.action_temperature *= self.action_temperature_decay
        self.random_exploration_temperature *= self.random_exploration_temperature_decay
        # Return the full AAS policy estimation
        return aas_policy_estimation, max_p_node
