from aas import config as cfg
from aas.state import OperationState
from aas.action import Action, Parallelization, Vectorization, NoTransformation
from typing import Literal, Optional
import math
import random


class Node:
    """Class to represent a node in the MCTS tree"""

    state: OperationState
    """The state of the operation at this node"""
    parent: Optional['Node']
    """The parent node of this node"""
    children: list['Node']
    """The children nodes of this node"""
    nb_visits: int
    """The number of times this node has been visited"""
    q: float
    """The max expected speedup of this node if mode is 'VEMS' or the average expected speedup if mode is 'VEAS'."""
    node_p: float
    """The probability of selecting this node from its parent node."""

    def __init__(self, state: OperationState, node_p: float = 1.0, parent: Optional['Node'] = None):
        """Initialize a new node in the MCTS tree.

        Args:
            state (OperationState): The state of the operation at this node.
            node_p (float): The probability of selecting this node from its parent node. Defaults to 1.0.
            parent (Optional[Node]): The parent node of this node. Defaults to None.
        """
        self.state = state
        self.parent = parent
        self.children = []
        self.nb_visits = 0
        self.q = 0.0
        self.node_p = node_p

    def add_child(self, child_state: OperationState, child_node_p: float):
        """Add a child node to this node.

        Args:
            child_state (OperationState): The state of the operation at the child node.
            child_node_p (float): The probability of selecting the child node from this node.

        Returns:
            Node: The child node that was added.
        """
        # Create child node
        child = Node(
            state=child_state,
            node_p=child_node_p,
            parent=self
        )
        # Add child to children list
        self.children.append(child)
        return child

    def update(self, expected_speedup: float, mode: Literal['VEMS', 'VEAS'] = 'VEMS'):
        """Update the node with the actual speedup value in backprobagation after simulation.

        Args:
            expected_speedup (float): The expected speedup value to update the node with.
            mode (Literal['VEMS', 'VEAS']): The update mode. 'VEMS' for max expected speedup and 'VEAS' for average expected speedup. Defaults to 'VEMS'.
        """
        # Update number of visits
        self.nb_visits += 1
        # Get Q value
        if mode == 'VEMS':
            self.q = max(self.q, expected_speedup)
        elif mode == 'VEAS':
            self.q = (self.q * (self.nb_visits - 1) + expected_speedup) / self.nb_visits
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def update_leaf(self, new_value: float):
        """Update the s score of the node with a new value.

        Args:
            new_value (float): The new value to update the s score with.
            mode (Literal['VEMS', 'VEAS']): The update mode. 'VEMS' for max expected speedup and 'VEAS' for average expected speedup.
        """
        # If node is not a leaf, raise an exception
        if self.children:
            raise Exception("The current node is not a leaf as it contains children.")
        # Update Q value
        self.q = new_value

    def get_s_score(self, c_puct: float = 1.0):
        """Get the s score of the node.

        Args:
            c_puct (float): The exploration parameter for the PUCT formula. Defaults to 1.0.

        Returns:
            float: The s score of the node.
        """
        if self.parent is not None:
            return self.q + c_puct * self.node_p * ((math.sqrt(self.parent.nb_visits) / (1 + self.nb_visits)))
        else:
            return self.q

    def get_available_actions(self) -> list[Action]:
        """Get the available actions from this node.

        Returns:
            list[Action]: The list of available actions.
        """
        transformation_names = [action.name for action in self.state.transformation_history]
        if self.is_terminal():
            # If the node is terminal, return an empty list
            return []
        elif Parallelization.DEFAULT_NAME in transformation_names:
            # If parallelization is already applied, return vectorization if possible or no transformation
            return [NoTransformation()] + ([Vectorization()] if Vectorization.is_possible(self.state.operation_features) else [])
        else:  # NOTE: This should only happen if transformation list is empty
            # Get tiling candidates and number of possible combinations
            nb_combinations = 1
            candidates = Parallelization.get_tiling_candidates(self.state.operation_features)
            for sub_candidates in candidates:
                nb_combinations *= len(sub_candidates)
            # Generate tiling combinations
            combinations = []
            if cfg.mcts_max_num_tile_combinations < 0:
                combinations = Parallelization.generate_tiling_combinations(candidates)
            else:
                seen_combination_ids = set()
                for i in min(nb_combinations, cfg.mcts_max_num_tile_combinations):
                    combination_id = random.randint(0, nb_combinations - 1)
                    while combination_id in seen_combination_ids:
                        combination_id = random.randint(0, nb_combinations - 1)
                    seen_combination_ids.add(combination_id)
                    combination = []
                    tmp_comb_id = combination_id
                    for sub_candidates in reversed(candidates):
                        combination = sub_candidates[tmp_comb_id % len(sub_candidates)] + combination
                        tmp_comb_id //= len(sub_candidates)
                    combinations.append(combination)

            return [NoTransformation()] + [Parallelization(combination) for combination in combinations] + ([Vectorization()] if Vectorization.is_possible(self.state.operation_features) else [])

    def is_terminal(self):
        """Check if the node is a terminal node.

        Returns:
            bool: True if the node is terminal, False otherwise.
        """
        transformation_names = [action.name for action in self.state.transformation_history]
        return (Vectorization.DEFAULT_NAME in transformation_names) or (NoTransformation.DEFAULT_NAME in transformation_names)

    def __repr__(self):
        return f"<Node nb_visits={self.nb_visits} s={self.get_s_score()} q={self.q} p_node={self.node_p}>"
