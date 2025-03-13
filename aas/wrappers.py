from aas import config as cfg
from aas.node import Node
from aas.nn import AASNetwork, CrossEntropyLoss
from aas.action import Action, Parallelization, Vectorization, NoTransformation
from aas.state import OperationState
import torch
from utils.torch_utils import sample_from_dist
from typing import Optional, Literal


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

    def get_action_from_hierarchical_probs(self, mode: Literal['greedy', 'stochastic'] = 'stochastic'):
        """Get the action selected in a hierarchical manner given the estimation.

        Args:
            mode (Literal['greedy', 'stochastic']): The mode to select the action. Defaults to 'stochastic'.

        Returns:
            Action: The selected action."""
        # Disable gradients
        with torch.no_grad():
            # Select the action given selection probabilities
            select_id = sample_from_dist(self.select_probs) if mode == 'stochastic' else torch.argmax(self.select_probs).item()
            if select_id == Parallelization.ID:
                # Get tile sizes for parallelization
                tile_sizes = []
                for i in range(cfg.max_num_loops):
                    param_id = sample_from_dist(self.parallel_params_probs[i]) if mode == 'stochastic' else torch.argmax(self.parallel_params_probs[i]).item()
                    tile_size = Parallelization.get_tile_size(param_id)
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


class AASNetworkWrapper:
    """Class to represent the AlphaAutoScheduler network wrapper. It uses the AlphaAutoScheduler network model and
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
        self.value_loss = torch.nn.MSELoss()
        # Define the optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.learning_rate)
        # Set stats
        self.stats = AASNetworkManagerStats()

    def train(self, data: list[tuple[OperationState, AASNetworkEstimation]]):
        """Train the AlphaAutoScheduler network on a given history data.

        Args:
            data (list[tuple[OperationState, AASNetworkEstimation]]): The history data to train the network.
        """
        # Reset stats if logging is enabled
        if cfg.logging:
            self.stats.reset()
        # Gather data
        input_arr = []
        policy_mask_arr = []
        parallel_params_mask_arr = []
        select_probs_target_arr = []
        parallel_params_probs_target_arr = []
        value_target_arr = []
        for state, target in data:
            # Get inputs
            input_arr.append(state.to_tensor())
            # Get masks
            policy_mask_arr.append(0.0 if target.policy.is_no_action_estimation() else 1.0)
            parallel_params_mask_arr.append(self.get_action_mask(state, target.policy))
            # Get target tensors
            select_probs_target_arr.append(target.policy.select_probs)
            parallel_params_probs_target_arr.append(target.policy.parallel_params_probs)
            value_target_arr.append(target.value)
        # Stack tensors
        input_tensors = torch.stack(input_arr)
        policy_mask = torch.tensor(policy_mask_arr)
        parallel_params_mask = torch.stack(parallel_params_mask_arr)
        select_probs_target = torch.stack(select_probs_target_arr)
        parallel_params_probs_target = torch.stack(parallel_params_probs_target_arr)
        value_target = torch.tensor(value_target_arr)
        # Calculate nb steps per epoch
        data_size = len(data)
        # nb_steps = data_size // cfg.batch_size + (1 if data_size % cfg.batch_size != 0 else 0)
        nb_steps = data_size // cfg.batch_size
        for _ in range(cfg.epochs):
            # Shuffle tensors
            perm = torch.randperm(len(data))
            input_tensors = input_tensors[perm]
            policy_mask = policy_mask[perm]
            parallel_params_mask = parallel_params_mask[perm]
            select_probs_target = select_probs_target[perm]
            parallel_params_probs_target = parallel_params_probs_target[perm]
            value_target = value_target[perm]
            # Train the network for one epoch
            for j in range(nb_steps):
                # Get batch
                batch_start = j * cfg.batch_size
                batch_end = (j + 1) * cfg.batch_size
                input_tensors_batch = input_tensors[batch_start:batch_end]
                policy_mask_batch = policy_mask[batch_start:batch_end]
                parallel_params_mask_batch = parallel_params_mask[batch_start:batch_end]
                select_probs_target_batch = select_probs_target[batch_start:batch_end]
                parallel_params_probs_target_batch = parallel_params_probs_target[batch_start:batch_end]
                value_target_batch = value_target[batch_start:batch_end]
                # Make a forward pass (train mode)
                select_probs_pred, parallel_params_probs_pred, value_pred = self.model(input_tensors_batch)
                # Reset gradients
                self.optimizer.zero_grad()
                # Calculate losses
                sl = self.ce_loss(select_probs_pred, select_probs_target_batch, mask=policy_mask_batch)
                ppls = torch.concatenate([self.ce_loss(parallel_params_probs_pred[:, i, :], parallel_params_probs_target_batch[:, i, :], mask=parallel_params_mask_batch[:, i]).unsqueeze(0) for i in range(cfg.max_num_loops)])
                vl = self.value_loss(value_pred, value_target_batch)
                # Save losses for stats
                if cfg.logging:
                    self.stats.selection_loss.append(sl.item())
                    for i in range(cfg.max_num_loops):
                        self.stats.parallel_params_loss[i].append(ppls[i].item())
                    self.stats.value_loss.append(vl.item())
                # Backward pass
                loss = sl + torch.sum(ppls) + vl
                loss.backward()
                # Optimize parameters
                self.optimizer.step()
            # Save stats
            if cfg.logging:
                self.stats.selection_loss.append(sl.item())
                for i in range(cfg.max_num_loops):
                    self.stats.parallel_params_loss[i].append(ppls[i].item())
                self.stats.value_loss.append(vl.item())

    def get_action_mask(self, state: OperationState, policy: AASNetworkPolicyEstimation):
        """Get the mask for target policy in given state.

        Args:
            state (OperationState): The operation state
            policy (AASNetworkPolicyEstimation): The policy estimation.

        Returns:
            torch.Tensor: The mask for parallel tile sizes selection.
        """
        # Set a mask for loop tile sizes selection
        parallel_params_mask = torch.zeros(cfg.max_num_loops)
        # If parallelization is not selected, return mask with zeros
        if policy.select_probs[Parallelization.ID] == 0:
            return parallel_params_mask
        else:  # Otherwise, mask loops which tile sizes are not needed
            nb_loops = len(state.operation_features.nested_loops)
            for i in range(cfg.max_num_loops):
                parallel_params_mask[i] = 1 if i < nb_loops else 0
        # Return masks
        return parallel_params_mask

    def get_action_prob(self, node: Node, action: Action, aas_policy_estimation: Optional[AASNetworkPolicyEstimation] = None):
        """Get the probability of an action given the curent node and the AASNetwork estimation.

        Args:
            node (Node): The current node just before performing the action.
            action (Action): The action to calculate the probability of.
            aas_estimation (Optional[AASNetworkPolicyEstimation]): The AASNetwork policy estimation. Defaults to None.
            If None, the network model is used to get the estimation.

        Returns:
            float: The probability of an action given the current node and the AASNetwork estimation.
        """
        # Get the action probabilities of the node
        if aas_policy_estimation is None:
            aas_policy_estimation = self.eval_node_policy(node)
        # Get probability of the node
        if isinstance(action, Parallelization):
            action_prob = aas_policy_estimation.select_probs[Parallelization.ID].item()
            for i, param in enumerate(action.params):
                param_idx = Parallelization.get_param_id(param)
                action_prob *= aas_policy_estimation.parallel_params_probs[i, param_idx].item()
        elif isinstance(action, Vectorization):
            action_prob = aas_policy_estimation.select_probs[Vectorization.ID].item()
        elif isinstance(action, NoTransformation):
            action_prob = aas_policy_estimation.select_probs[NoTransformation.ID].item()
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
        # Get the next action probabilities of the node
        with torch.no_grad():
            # Set model to evaluation mode
            self.model.eval()
            # Make prediction
            select_probs, parallel_params_probs, value = self.model(node.state.to_tensor().unsqueeze(0))
            select_probs = select_probs.squeeze(0)
            parallel_params_probs = parallel_params_probs.squeeze(0)
            value = value.squeeze(0)
            # Set model back to training mode
            self.model.train()
        # Create the AASNetwork estimation
        aas_estimation = AASNetworkEstimation(
            policy=AASNetworkPolicyEstimation(
                select_probs=select_probs,
                parallel_params_probs=parallel_params_probs
            ),
            value=value
        )
        # Return the action probabilities
        return aas_estimation

    def eval_node_policy(self, node: Node) -> AASNetworkPolicyEstimation:
        """Evaluate the policy network on a node.

        Args:
            node (Node): The node to evaluate.

        Returns:
            AASNetworkPolicyEstimation: The AASNetwork policy estimation.
        """
        # Get the next action probabilities of the node
        with torch.no_grad():
            # Set model to evaluation mode
            self.model.eval()
            # Make prediction
            select_probs, parallel_params_probs = self.model.eval_policy(node.state.to_tensor().unsqueeze(0))
            select_probs = select_probs.squeeze(0)
            parallel_params_probs = parallel_params_probs.squeeze(0)
            # Set model back to training mode
            self.model.train()
        # Create the AASNetwork policy estimation
        aas_policy_estimation = AASNetworkPolicyEstimation(
            select_probs=select_probs,
            parallel_params_probs=parallel_params_probs
        )
        # Return the action probabilities
        return aas_policy_estimation

    def eval_node_value(self, node: Node) -> float:
        """Evaluate the value network on a node.

        Args:
            node (Node): The node to evaluate.

        Returns:
            float: The value of the node.
        """
        # Get the next action probabilities of the node
        with torch.no_grad():
            # Set model to evaluation mode
            self.model.eval()
            # Make prediction
            value = self.model.eval_value(node.state.to_tensor().unsqueeze(0))
            value = value.squeeze(0)
            # Set model back to training mode
            self.model.train()
        # Return the value
        return value.item()

    def eval(self, state: OperationState) -> AASNetworkEstimation:
        """Evaluate the policy network and value network on a node.

        Args:
            state (OperationState): The state to evaluate.

        Returns:
            AASNetworkEstimation: The AASNetwork estimation.
        """
        # Get the next action probabilities of the node
        with torch.no_grad():
            # Set model to evaluation mode
            self.model.eval()
            # Make prediction
            select_probs, parallel_params_probs, value = self.model(state.to_tensor().unsqueeze(0))
            select_probs = select_probs.squeeze(0)
            parallel_params_probs = parallel_params_probs.squeeze(0)
            value = value.squeeze(0)
            # Set model back to training mode
            self.model.train()
        # Apply masks
        # If state is terminal, return no action AAS policy estimation and state value
        if state.is_terminal():
            return AASNetworkEstimation(
                policy=self.get_no_action_aas_policy_estimation(),
                value=value
            )
        transformation_names = [action.name for action in state.transformation_history]
        op_features = state.operation_features
        parallelization_applied = Parallelization.DEFAULT_NAME in transformation_names
        # If parallelization is already applied don't apply it again
        if parallelization_applied:
            parallel_action = next(action for action in state.transformation_history if isinstance(action, Parallelization))
            op_features = parallel_action.update_op_features(state.operation_features)
            select_probs[Parallelization.ID] = 0.0
        else:
            select_probs[NoTransformation.ID] = 0.0
        # If vectorization is not possible, don't apply it
        if not Vectorization.is_possible(op_features):
            select_probs[Vectorization.ID] = 0.0
        # If parallelization is already applied, mask all parallelization parameters
        if parallelization_applied:
            parallel_params_probs[:, :] = 0.0
            parallel_params_probs[:, 0] = 1.0
        else:  # Otherwise, mask parallelization parameters that don't divide the loop size
            for i, loop in enumerate(op_features.nested_loops):
                for j in range(cfg.num_tile_sizes + 1):
                    tile_size = Parallelization.get_tile_size(j)
                    if tile_size > 0 and loop.upper_bound % tile_size != 0:
                        parallel_params_probs[i, j] = 0.0
                # Normalize the probabilities
                loop_probs_sum = parallel_params_probs[i].sum()
                if loop_probs_sum > 0:
                    parallel_params_probs[i] /= loop_probs_sum
                else:
                    parallel_params_probs[i, 0] = 1.0
            # Mask parallelization parameters for loops that are not present in the operation
            nb_loops = len(op_features.nested_loops)
            parallel_params_probs[nb_loops:, :] = 0.0
            parallel_params_probs[nb_loops:, 0] = 1.0
        # Normalize selection probabilities
        select_probs_sum = select_probs.sum()
        if select_probs_sum > 0:
            select_probs /= select_probs_sum
        else:
            select_probs[NoTransformation.ID] = 1.0

        # Create the AASNetwork estimation
        aas_estimation = AASNetworkEstimation(
            policy=AASNetworkPolicyEstimation(
                select_probs=select_probs,
                parallel_params_probs=parallel_params_probs
            ),
            value=value
        )
        # Return the action probabilities
        return aas_estimation

    def evaluate_tree(self, root: Node, temperature: float, mode: Literal['greedy', 'stochastic'] = 'stochastic'):
        """Get the full AASNetwork policy estimation and the action selected given MCTS probabilities after the root node.

        Args:
            node (Node): The node to map the children from.
            temperature (float): The temperature parameter for the MCTS probabilities.
            mode (Literal['greedy', 'stochastic']): The mode to select the action. Defaults to 'stochastic'.

        Returns:
            AASNetworkPolicyEstimation: The full AASNetwork policy estimation.
            Node: The child node selected MCTS probabilities. The root is returned if no children.
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
        max_prob_action = aas_policy_estimation.get_action_from_hierarchical_probs(mode=mode)
        selected_node = children_dict[str(max_prob_action)]
        # Return results
        return aas_policy_estimation, selected_node

    def get_no_action_aas_policy_estimation(self):
        """Get the AASNetwork policy estimation for no transformation action.

        Returns:
            AASNetworkPolicyEstimation: The AASNetwork policy estimation for no transformation action.
        """
        # Return the AASNetwork policy estimation
        return AASNetworkPolicyEstimation.no_action_estimation()
