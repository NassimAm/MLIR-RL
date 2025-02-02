from aas import config as cfg
from aas.node import Node
from aas.nn import AASNetwork, BSELoss, CrossEntropyLoss
from aas.action import Action, Parallelization, Vectorization, NoTransformation
from aas.state import OperationState
from aas.observation.benchmark import BenchmarkFeatures
import torch
from utils.torch_utils import sample_from_dist
from typing import Optional
import math


class AASNetworkPolicyEstimation:
    """Class to represent an policy estimation that AlphaAutoScheduler network would make."""

    select_probs: torch.Tensor
    """Probabilities predicted by the select network."""
    parallel_params_probs: torch.Tensor
    """Probabilities predicted by the parallelization parameters network."""

    def __init__(self, select_probs: torch.Tensor, parallel_params_probs: torch.Tensor):
        """Initialize the AAS network policy estimation.

        Args:
            select_probs (torch.Tensor): probabilities predicted by the select network.
            parallel_params_logits (torch.Tensor): probabilities predicted by the parallelization parameters network.
        """
        self.select_probs = select_probs
        self.parallel_params_probs = parallel_params_probs

    def get_max_hierarchical_prob_action(self):
        """Get the action with the highest probability in a hierarchical manner given the estimation.

        Returns:
            Action: The action with the highest probability."""
        # Disable gradients
        with torch.no_grad():
            # Sample the action with the given selection probabilities
            select_id = sample_from_dist(self.select_probs)
            if select_id == Parallelization.ID:
                # Get tile sizes for parallelization
                tile_sizes = []
                for i in range(cfg.max_num_loops):
                    parallel_id = sample_from_dist(self.parallel_params_probs[i])
                    tile_size = 2 ** (parallel_id - 1) if parallel_id > 0 else 0
                    tile_sizes.append(tile_size)
                return Parallelization(tile_sizes)
            elif select_id == Vectorization.ID:
                return Vectorization()
            else:
                return NoTransformation()

    def no_action_estimation():
        """Get the AAS network policy estimation for no transformation action.

        Returns:
            AASNetworkPolicyEstimation: The AAS network policy estimation for no transformation action.
        """
        # Return the AAS network policy estimation
        return AASNetworkPolicyEstimation(
            select_probs=torch.zeros(cfg.num_transformations),
            parallel_params_probs=torch.zeros((cfg.max_num_loops, cfg.num_tile_sizes + 1))
        )

    def is_no_action_estimation(self):
        """Check if the policy estimation is for no transformation action.

        Returns:
            bool: True if the policy estimation is for no transformation action, False otherwise.
        """
        return (self.select_probs.sum() == 0).item()

    def __repr__(self):
        """Get the string representation of the AAS network policy estimation."""
        return (f'Select probs:\n{self.select_probs}\n'
                f'Parallel params probs:\n{self.parallel_params_probs}')


class AASNetworkEstimation:
    """Class to represent an estimation that AlphaAutoScheduler network would make."""

    policy: AASNetworkPolicyEstimation
    """The policy estimation of the operation."""
    value: torch.Tensor
    """The value of the operation."""

    def __init__(self, policy: AASNetworkPolicyEstimation, value: torch.Tensor):
        """Initialize the AAS network estimation.

        Args:
            policy (AASNetworkPolicyEstimation): The policy estimation of the operation.
            value (torch.Tensor): The value of the operation.
        """
        self.policy = policy
        self.value = value

    def get_value(self):
        """Get the value of the operation."""
        return self.value

    def __repr__(self):
        """Get the string representation of the AAS network estimation."""
        return (f'Policy:\n{self.policy}\n'
                f'Value: {self.value}')


class AASNetworkManagerStats:
    """The AlphaAutoScheduler Stats class. It contains all collected stats during training if logging is enabled."""

    selection_loss: list[float]
    """The selection loss history."""
    parallel_params_loss: list[list[float]]
    """The parallelization parameters loss history."""
    value_loss: list[float]
    """The value loss history."""

    def __init__(self):
        """Initialize the AlphaAutoScheduler stats."""
        self.reset()

    def reset(self):
        """Reset the stats."""
        self.selection_loss = []
        self.parallel_params_loss = [[] for _ in range(cfg.max_num_loops)]
        self.value_loss = []


class AASNetworkManager:
    """Class to represent the AlphaAutoScheduler estimator. It uses the AlphaAutoScheduler network model and
    converts its outputs to environment needs."""

    model: AASNetwork
    """The AlphaAutoScheduler network model."""

    def __init__(self, model: AASNetwork):
        """Initialize the policy estimator.

        Args:
            model (AASNetwork): The AlphaAutoScheduler network model
        """
        # Set the model
        self.model = model
        # Define losses
        self.ce_loss = CrossEntropyLoss()
        if cfg.mcts_estimation_mode == 'VEMS':
            self.value_loss = BSELoss()
        else:
            self.value_loss = torch.nn.MSELoss()
        # Define the optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.learning_rate)
        # Set stats
        self.stats = AASNetworkManagerStats()

    def train_on_trajectory(self, trajectory: list[tuple[OperationState, AASNetworkEstimation]]):
        """Train the AlphaAutoScheduler network on a given trajectory.

        Args:
            trajectory (list[tuple[OperationState, AASNetworkEstimation]]): The trajectory to train the AlphaAutoScheduler network on.
        """
        # Reset stats if logging is enabled
        if cfg.logging:
            self.stats.reset()
        # Train the model on the trajectory
        for state, target in trajectory:
            # Get input tensor
            x = state.to_tensor()
            latest_action = state.transformation_history[-1] if len(state.transformation_history) > 0 else None
            policy_mask = 0.0 if target.policy.is_no_action_estimation() else 1.0
            parallel_params_mask = self.get_action_mask(state, latest_action)
            # Make a forward pass (train mode)
            select_probs_pred, parallel_params_probs_pred, value_pred = self.model(x)
            select_probs_pred = select_probs_pred.unsqueeze(0)
            parallel_params_probs_pred = parallel_params_probs_pred.unsqueeze(0)
            value_pred = value_pred.unsqueeze(0)
            value_pred.retain_grad()
            # Get target tensors
            select_probs_target = target.policy.select_probs.unsqueeze(0)
            parallel_params_probs_target = target.policy.parallel_params_probs.unsqueeze(0)
            value_target = target.value.unsqueeze(0)
            # Reset gradients
            self.optimizer.zero_grad()
            # Calculate losses
            sl = self.ce_loss(select_probs_pred, select_probs_target)
            ppls = torch.concatenate([self.ce_loss(parallel_params_probs_pred[:, i, :], parallel_params_probs_target[:, i, :] * parallel_params_mask[i]).unsqueeze(0) for i in range(cfg.max_num_loops)])
            print("Predicted Value: ", value_pred)
            print("Target Value: ", value_target)
            vl = self.value_loss(value_pred, value_target)
            print("Value Loss: ", vl)
            # Save losses for stats
            if cfg.logging:
                self.stats.selection_loss.append(sl.item())
                for i in range(cfg.max_num_loops):
                    self.stats.parallel_params_loss[i].append(ppls[i].item())
                self.stats.value_loss.append(vl.item())
            # Backward pass
            loss = (sl + torch.sum(ppls)) * policy_mask + vl
            loss.backward()
            print("Value Grad: ", value_pred.grad)
            # Optimize parameters
            self.optimizer.step()

    def get_action_mask(self, state: OperationState, action: Optional[Action]):
        """Get the mask for the action.

        Args:
            state (OperationState): The current state.
            action (Optional[Action]): The action to get the mask for.

        Returns:
            torch.Tensor: The mask for parallel tile sizes selection.
        """
        # Set a mask for loop tile sizes selection
        parallel_params_mask = torch.zeros(cfg.max_num_loops)
        # If no action is given, return masks
        if action is None:
            return parallel_params_mask
        # Otherwise, set masks for the action
        if isinstance(action, Parallelization):
            # Mask loops which tile sizes are not needed
            nb_loops = len(state.operation_features.nested_loops)
            for i in range(cfg.max_num_loops):
                parallel_params_mask[i] = 1 if i < nb_loops else 0
        # Return masks
        return parallel_params_mask

    def get_action_prob(self, node: Node, action: Action, aas_estimation: Optional[AASNetworkEstimation] = None):
        """Get the probability of an action given the curent node and the AASNetwork estimation.

        Args:
            node (Node): The current node just before performing the action.
            action (Action): The action to calculate the probability of.
            aas_estimation (Optional[AASNetworkEstimation]): The AASNetwork estimation. Defaults to None.
            If None, the network model is used to get the estimation.

        Returns:
            float: The probability of an action given the current node and the AASNetwork estimation.
        """
        # Get the action probabilities of the node
        if aas_estimation is None:
            with torch.no_grad():
                select_probs, parallel_params_probs, value = self.model(node.state.to_tensor())
                aas_estimation = AASNetworkEstimation(
                    policy=AASNetworkPolicyEstimation(
                        select_probs=select_probs,
                        parallel_params_probs=parallel_params_probs
                    ),
                    value=value
                )
        # Get probability of the node
        if isinstance(action, Parallelization):
            action_prob = aas_estimation.policy.select_probs[Parallelization.ID].item()
            for i, param in enumerate(action.params):
                param_idx = Parallelization.get_param_id(param)
                action_prob *= aas_estimation.policy.parallel_params_probs[i, param_idx].item()
        elif isinstance(action, Vectorization):
            action_prob = aas_estimation.policy.select_probs[Vectorization.ID].item()
        elif isinstance(action, NoTransformation):
            action_prob = aas_estimation.policy.select_probs[NoTransformation.ID].item()
        else:
            raise ValueError(f'Action {action} is not supported !')

        return action_prob

    def eval_node(self, node: Node) -> AASNetworkEstimation:
        """Evaluate the policy network and value network on a node.

        Args:
            node (Node): The node to evaluate.

        Returns:
            AASNetworkEstimation: The AASNetwork estimation.
        """
        # Set model to evaluation mode
        self.model.eval()
        # Get the next action probabilities of the node
        with torch.no_grad():
            select_probs, parallel_params_probs, value = self.model(node.state.to_tensor())
        # Create the AASNetwork estimation
        aas_estimation = AASNetworkEstimation(
            policy=AASNetworkPolicyEstimation(
                select_probs=select_probs,
                parallel_params_probs=parallel_params_probs
            ),
            value=value
        )
        # Set model back to training mode
        self.model.train()
        # Return the action probabilities
        return aas_estimation

    def evaluate_tree(self, root: Node, temperature: float):
        """Get the full AASNetwork policy estimation and the action with the highest MCTS probability after the root node.

        Args:
            node (Node): The node to map the children from.
            temperature (float): The temperature parameter for the MCTS probabilities.

        Returns:
            AASNetworkPolicyEstimation: The full AASNetwork policy estimation.
            Node: The child node with the highest MCTS probability. The root is returned if no children.
        """
        # If node has no children, return no action AAS policy estimation
        if not root.children:
            return self.get_no_action_aas_policy_estimation(), root
        # Disable gradients
        with torch.no_grad():
            # Initialize selection probabilities over transformations
            select_probs = torch.zeros(cfg.num_transformations)
            # Initialize parallelization parameters probabilities
            parallel_params_probs = torch.zeros((cfg.max_num_loops, cfg.num_tile_sizes + 1))
            parallel_prob = torch.tensor(0.0)
            # Calculate the denominator for MCTS next action probabilities
            denominator = sum([child.nb_visits ** (1 / temperature) for child in root.children])
            # Save chidren nodes in a dict for easy access later
            # The dict would have string representation of the action taken from root to get to that node as a key
            # and the chid node as a value
            children_dict = {}
            # For each child node get the latest action and its MCTS probability
            for i, child in enumerate(root.children):
                child_action = child.state.transformation_history[-1]
                child_p = child.nb_visits ** (1 / temperature) / denominator
                # Save child in dict
                children_dict[str(child_action)] = child
                # Put the probability in the right place in the probability tensor
                if isinstance(child_action, Parallelization):
                    # Calculate parallelization selection probability
                    parallel_prob += child_p
                    # For parallelization, the marginal probability is calculated instead of using MCTS joint probability over tiling sizes
                    for i, param in enumerate(child_action.params):
                        param_idx = Parallelization.get_param_id(param)
                        parallel_params_probs[i, param_idx] += child_p
                elif isinstance(child_action, Vectorization):
                    select_probs[Vectorization.ID] = child_p
                elif isinstance(child_action, NoTransformation):
                    select_probs[NoTransformation.ID] = child_p
                else:
                    raise ValueError(f'Action {child_action} is not supported !')
            # Correct parallelization parameters probabilities by calculating conditional probabilities
            parallel_prob_by_loop = parallel_params_probs.sum(dim=1)
            select_probs[Parallelization.ID] = parallel_prob
            for i in range(cfg.max_num_loops):
                if parallel_prob_by_loop[i] > 0:
                    # If the prior probability of select the loop is not zero, divide by that probability to get the conditional probability
                    parallel_params_probs[i] /= parallel_prob_by_loop[i]
                else:
                    # If the prior probability of select the loop is zero, set probability of not tiling to 1
                    parallel_params_probs[i, 0] = 1.0
        # Create the AASNetwork policy estimation
        aas_policy_estimation = AASNetworkPolicyEstimation(
            select_probs=select_probs,
            parallel_params_probs=parallel_params_probs
        )
        # Get the child node with the highest MCTS probability
        max_prob_action = aas_policy_estimation.get_max_hierarchical_prob_action()
        max_prob_node = children_dict[str(max_prob_action)]
        # Return results
        return aas_policy_estimation, max_prob_node

    def get_no_action_aas_policy_estimation(self):
        """Get the AASNetwork policy estimation for no transformation action.

        Returns:
            AASNetworkPolicyEstimation: The AASNetwork policy estimation for no transformation action.
        """
        # Return the AASNetwork policy estimation
        return AASNetworkPolicyEstimation.no_action_estimation()

    def get_speedup_reward(self, bench_features: BenchmarkFeatures, exec_time: int):
        """Get the speedup reward based on the execution time.

        Args:
            bench_features (BenchmarkFeatures): The benchmark features.
            exec_time (int): The execution time of the current node.

        Returns:
            float: The speedup reward.
        """
        root_exec_time = bench_features.exec_time
        return min(4, max(-4, math.log(root_exec_time / exec_time, 10)))
        # return math.log(root_exec_time / exec_time, 2)
        # return root_exec_time / exec_time
