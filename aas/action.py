from aas import config as cfg
from aas.observation import OperationFeatures


class Action:
    """Class to represent a transformation as an agent action."""
    _name: str
    """The name of the transformation"""

    def __init__(self, name: str):
        """Initialize a new action.

        Args:
            name (str): The name of the transformation.
        """
        self._name = name

    @property
    def name(self):
        """The name of the transformation."""
        return self._name

    def __repr__(self):
        return f'{self.name}()'


class ParameterizedAction(Action):
    """Class to represent a parameterized transformation as an agent action."""
    params: list[int]
    """The parameters of the transformation"""

    def __init__(self, name: str, params: list[int]):
        """Initialize a new parameterized action.

        Args:
            name (str): The name of the transformation.
            params (list[int]): The parameters of the transformation.
        """
        super().__init__(name)
        self.params = params

    def __repr__(self):
        return f'{self.name}({', '.join([str(e) for e in self.params])})'


class Parallelization(ParameterizedAction):
    """Class to represent a parallelization transformation as an agent action."""

    DEFAULT_NAME = 'TP'
    """The default name of the parallelization transformation"""
    ID = 0
    """The ID of the parallelization transformation"""

    def __init__(self, params: list[int]):
        """Initialize a new parallelization action.

        Args:
            params (list[int]): The parameters of the transformation.
        """
        super().__init__(Parallelization.DEFAULT_NAME, params)

    def generate_tiling_combinations(candidates: list[list[int]]):
        """Generate all possible tiling combinations from the list of candidates.

        Args:
            candidates (list[list[int]]): The list of candidates to generate tiling combinations from.

        Returns:
            list[list[int]]: The list of all possible tiling combinations.
        """
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return [[candidate] for candidate in candidates[0]]
        else:
            sub_candidates = Parallelization.generate_tiling_combinations(candidates[1:])
            return [[candidate] + sub_candidate for candidate in candidates[0] for sub_candidate in sub_candidates]

    def get_tiling_candidates(operation_features: OperationFeatures):
        """Get the tiling candidates for the operation features.

        Args:
            operation_features (OperationFeatures): The operation features to get the tiling candidates from.

        Returns:
            list[list[int]]: The list of tiling candidates.
        """
        candidates = []
        for nested_loop in operation_features.nested_loops:
            # If upperbound equal 1, we only have candidates of 1
            if nested_loop.upper_bound == 1:
                sub_candidates = [0, 1]
            # If the data format is json and the iterator type is reduction, we don't do tiling
            # TODO: the condition has to change because it's not related to the data format, it's related to a non thread safe tiling problem
            # so we skip it to not let it happen for now
            elif cfg.data_format == 'json' and nested_loop.iterator_type == 'reduction':
                sub_candidates = [0]
            else:
                # We take the divisors of the upperbound
                sub_candidates = [0]
                i = 1
                while i <= nested_loop.upper_bound and len(sub_candidates) < cfg.num_tile_sizes:
                    if nested_loop.upper_bound % i == 0:
                        sub_candidates.append(i)
                    i *= 2
            candidates.append(sub_candidates)
        # Fill other loops with 0 as candidate
        for _ in range(len(operation_features.nested_loops), cfg.max_num_loops):
            candidates.append([0])

        return candidates


class Vectorization(Action):
    """Class to represent a vectorization transformation as an agent action."""

    DEFAULT_NAME = 'V'
    """The default name of the vectorization transformation"""
    ID = 1
    """The ID of the vectorization transformation"""

    def __init__(self):
        """Initialize a new vectorization action."""
        super().__init__(Vectorization.DEFAULT_NAME)

    def is_possible(operation_features: OperationFeatures):
        return operation_features.op_iter_space_size <= cfg.vect_size_limit


class NoTransformation(Action):
    """Class to represent a no transformation action as an agent action."""

    DEFAULT_NAME = 'NT'
    """The default name of the no transformation action"""
    ID = 2
    """The ID of the no transformation action"""

    def __init__(self):
        """Initialize a new no transformation action."""
        super().__init__(NoTransformation.DEFAULT_NAME)
