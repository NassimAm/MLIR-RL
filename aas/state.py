from aas import config as cfg
from aas.observation import OperationFeatures, formula_str_to_list
from aas.action import Action, Parallelization
import torch
import math


class OperationState:
    """Class to represent the operation state."""
    bench_name: str
    """The benchmark's name."""
    operation_tag: str
    """Tag used to identify the operation in the MLIR code."""
    operation_features: OperationFeatures
    """Features of the operation."""
    transformed_code: str
    """The operation string with wrapping and transformations."""
    step_count: int
    """The current step in the list of transformations applied to the operation."""
    transformation_history: list[Action]
    """List of transformations with their parameters applied to the operation."""

    def __init__(self, bench_name: str, operation_tag: str, operation_features: OperationFeatures,
                 transformed_code: str, step_count: int, transformation_history: list[Action]):
        """Initialize the operation state."""
        self.bench_name = bench_name
        self.operation_tag = operation_tag
        self.operation_features = operation_features
        self.transformed_code = transformed_code
        self.step_count = step_count
        self.transformation_history = transformation_history

    def get_tensor_dim():
        """Get the tensor dimension of the operation state.

        Returns:
            int: The observation tensor dimension.
        """
        # TODO: Update this when adding new transformations other than parallelization
        L = cfg.max_num_loops
        SL = cfg.max_num_stores_loads
        LSD = cfg.max_num_load_store_dim
        T = cfg.num_tile_sizes
        return 6 + 5 + 1 + L * 2 + SL * LSD * L + LSD * L + L * (T + 1)

    def to_tensor(self):
        """Convert the operation state to a torch tensor.

        Returns:
            torch.Tensor: The operation state as a torch tensor.
        """

        # Disable gradient computation
        with torch.no_grad():
            # Operation type (size=6)
            operation_type_ohe = torch.zeros(6)
            if self.operation_features.operation_type == 'matmul':
                operation_type_ohe[0] = 1
            elif 'conv_2d' in self.operation_features.operation_type:
                operation_type_ohe[1] = 1
            elif self.operation_features.operation_type == 'pooling':
                operation_type_ohe[2] = 1
            elif self.operation_features.operation_type == 'add':
                operation_type_ohe[3] = 1
            elif self.operation_features.operation_type == 'generic':
                operation_type_ohe[4] = 1
            else:
                operation_type_ohe[5] = 1

            # Operations count (size=5)
            operations_count = torch.tensor(list(self.operation_features.op_count.values()))

            # Operation iteration space size (size = 1)
            op_iter_space_size = torch.tensor([self.operation_features.op_iter_space_size])
            if cfg.normalize_features:
                op_iter_space_size = torch.log2(op_iter_space_size)

            # Nested loop features: (upper bound, iter type) (size = max_num_loops * 2)
            indices = [nested_loop.arg for nested_loop in self.operation_features.nested_loops]
            indices_dim = {arg: i for (i, arg) in enumerate(indices)}
            nested_loops = torch.zeros((cfg.max_num_loops, 2))
            for i, nested_loop in enumerate(self.operation_features.nested_loops):
                if i == cfg.max_num_loops:
                    break
                nested_loops[i, 0] = math.log2(nested_loop.upper_bound) if cfg.normalize_features else nested_loop.upper_bound
                nested_loops[i, 1] = 1 if nested_loop.iterator_type == 'parallel' else 0

            # Load access matrices (size = max_num_stores_loads * max_num_load_store_dim * max_num_loops)
            load_data = self.operation_features.load_data
            load_access_matrices = torch.zeros((cfg.max_num_stores_loads, cfg.max_num_load_store_dim, cfg.max_num_loops))
            for load_i, load in enumerate(load_data):
                if load_i == cfg.max_num_stores_loads:
                    break
                dimensions_terms = [formula_str_to_list(term) for term in load]
                for m, dimension_term in enumerate(dimensions_terms):
                    for index, factor in dimension_term:
                        if index in indices_dim:
                            n = indices_dim[index]
                            load_access_matrices[load_i, m, n] = factor

            # Store access matrices (size = max_num_load_store_dim * max_num_loops)
            store_data = self.operation_features.store_data
            store_access_matrices = torch.zeros((cfg.max_num_load_store_dim, cfg.max_num_loops))
            dimensions_terms = [formula_str_to_list(term) for term in store_data]
            for m, dimension_term in enumerate(dimensions_terms):
                for index, factor in dimension_term:
                    n = indices_dim[index]
                    store_access_matrices[m, n] = factor

            # Action history (size = max_num_loops * (num_tile_sizes + 1))
            # TODO: Update this when adding new transformations other than parallelization
            action_history = torch.zeros((cfg.max_num_loops, cfg.num_tile_sizes + 1))
            for action in self.transformation_history:
                if isinstance(action, Parallelization):
                    for i, param in enumerate(action.params):
                        idx = int(math.log2(param)) + 1 if param > 0 else 0
                        action_history[i, idx] = 1
            # Reshape tensors if needed
            nested_loops = nested_loops.reshape(-1)
            load_access_matrices = load_access_matrices.reshape(-1)
            store_access_matrices = store_access_matrices.reshape(-1)
            action_history = action_history.reshape(-1)

            # Concatenate the feature vectors
            feature_vector = torch.concatenate([
                operation_type_ohe,
                operations_count,
                op_iter_space_size,
                nested_loops,
                load_access_matrices,
                store_access_matrices,
                action_history
            ])

        return feature_vector

    def transformation_history_to_str(self):
        """Convert the transformation history to a string.

        Returns:
            str: The transformation history as a string.
        """
        return ''.join([str(action) for action in self.transformation_history])
