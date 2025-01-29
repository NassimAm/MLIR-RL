from aas import config as cfg
from aas.wrappers import AASNetworkManager
from aas.node import Node
from typing import Literal


class MCTS:
    """Class to represent the Monte Carlo Tree Search algorithm"""

    aas_network_manager: AASNetworkManager
    """The AlphaAutoScheduler network manager to calculate the prior probabilities of the actions and node values."""
    mode: Literal['VEMS', 'VEAS']
    """The mode of the MCTS algorithm. Can be 'VEMS' or 'VEAS'"""
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
        self.mode = cfg.mcts_estimation_mode
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
            node = max(node.children, key=lambda x: x.get_s_score(self.c_puct))
        return node

    def expand(self, node: Node):
        """Expand a node in the MCTS tree by adding its children and evaluate it with value approximation.

        Args:
            node (Node): The node to expand.
        """
        # Evaluate the node
        # TODO: It would be useful if v is always an underestimation of the real speedup
        aas_estimation = self.aas_network_manager.eval_node(node)
        node.update_leaf(aas_estimation.get_value())
        # Get available actions
        available_actions = node.get_available_actions()
        for action in available_actions:
            # Get next state
            next_state = node.state.next(action)
            # Process child node exploration factor
            child_node_factor = self.aas_network_manager.get_action_exploration_factor(action, self.random_exploration_temperature, aas_estimation=aas_estimation)
            # Add child node to the tree
            node.add_child(next_state, child_node_factor)

        # print("Expanded node:", node)
        # print("Children:", [str(child) for child in node.children])

    def backpropagate(self, node: Node):
        """Backpropagate the speedup value up the MCTS tree.

        Args:
            node (Node): The node to backpropagate from.
        """
        while node is not None:
            node.update(node.q, self.mode)
            node = node.parent

    def run(self, root: Node, n_iterations: int):
        """Perform the MCTS search for a given number of iterations and convert
         action probabilities to AASNetwork policy estimation.

        Args:
            root (Node): The root node of the MCTS tree.
            n_iterations (int): The number of iterations to perform the search.

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
        aas_policy_estimation, max_p_node = self.aas_network_manager.evaluate_tree(root, self.action_temperature)
        # Update temperatures
        self.action_temperature *= self.action_temperature_decay
        self.random_exploration_temperature *= self.random_exploration_temperature_decay
        # Return the full AAS policy estimation
        return aas_policy_estimation, max_p_node
