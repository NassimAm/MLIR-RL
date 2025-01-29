from aas import config as cfg
from aas.node import Node
from aas.nn import AASNetwork, BSELoss, CrossEntropyLoss
from aas.action import Action, Parallelization, Vectorization, NoTransformation
from aas.state import OperationState
import torch
import math
from typing import Optional


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

    def get_action_prob(self, action: Action, random_action_temperature: float):
        """Get the probability of an action given the estimation.

        Args:
            action (Action): The action to get the probability of.
            random_action_temperature (float): The temperature parameter for the random action selection.

        Returns:
            float: The probability of the action."""

        if isinstance(action, Parallelization):
            parallel_prob = self.select_probs[Parallelization.ID].item()
            joint_action_prob = parallel_prob
            for i, param in enumerate(action.params):
                joint_action_prob *= self.parallel_params_probs[i, (int(math.log2(param)) + 1 if param > 0 else 0)].item()
            action_prob = random_action_temperature * parallel_prob + (1 - random_action_temperature) * joint_action_prob
        elif isinstance(action, NoTransformation):
            action_prob = self.select_probs[NoTransformation.ID].item()
        elif isinstance(action, Vectorization):
            action_prob = self.select_probs[Vectorization.ID].item()
        else:
            raise ValueError(f'Action {action} is not supported !')

        return action_prob

    def get_max_hierarchical_prob_action(self):
        """Get the action with the highest probability in a hierarchical manner given the estimation.

        Returns:
            Action: The action with the highest probability."""
        # Disable gradients
        with torch.no_grad():
            # Get the action with the highest selection probability
            max_select_id = torch.argmax(self.select_probs).item()
            if max_select_id == Parallelization.ID:
                # Get tile sizes for parallelization
                tile_sizes = []
                for i in range(cfg.max_num_loops):
                    max_parallel_id = torch.argmax(self.parallel_params_probs[i]).item()
                    max_tile_size = 2 ** (max_parallel_id - 1) if max_parallel_id > 0 else 0
                    tile_sizes.append(max_tile_size)
                return Parallelization(tile_sizes)
            elif max_select_id == Vectorization.ID:
                return Vectorization()
            else:
                return NoTransformation()


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

    def get_action_prob(self, action: Action, random_action_temperature: float):
        """Get the probability of an action given the estimation.

        Args:
            action (Action): The action to get the probability of.
            random_action_temperature (float): The temperature parameter for the random action selection.

        Returns:
            float: The probability of the action."""
        return self.policy.get_action_prob(action, random_action_temperature)

    def get_value(self):
        """Get the value of the operation."""
        return self.value


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
            # Make a forward pass (train mode)
            select_probs_pred, parallel_params_probs_pred, value_pred = self.model(x)
            select_probs_pred = select_probs_pred.unsqueeze(0)
            parallel_params_probs_pred = parallel_params_probs_pred.unsqueeze(0)
            value_pred = value_pred.unsqueeze(0)
            # Get target tensors
            select_probs_target = target.policy.select_probs.unsqueeze(0)
            parallel_params_probs_target = target.policy.parallel_params_probs.unsqueeze(0)
            value_target = target.value.unsqueeze(0)
            # Reset gradients
            self.optimizer.zero_grad()
            # Calculate losses
            print("Selection")
            print(select_probs_pred)
            print(select_probs_target)
            print("Parallel")
            print(parallel_params_probs_pred)
            print(parallel_params_probs_target)
            print("Value")
            print(value_pred)
            print(value_target)
            sl = self.ce_loss(select_probs_pred, select_probs_target)
            ppls = torch.concatenate([self.ce_loss(parallel_params_probs_pred[:, i, :], parallel_params_probs_target[:, i, :]).unsqueeze(0) for i in range(cfg.max_num_loops)])
            vl = self.value_loss(value_pred, value_target)
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

    def get_action_prob(self, action: Action, random_action_temperature: float, aas_estimation: Optional[AASNetworkEstimation] = None):
        """Get the probability of an action given the action probabilities.

        Args:
            action (Action): The action to get the probability of.
            random_action_temperature (float): The temperature parameter for the random action selection.
            aas_estimation (Optional[AASNetworkEstimation]): The estimation made by the AASNetwork. Defaults to None.

        Returns:
            float: The probability of the action.
        """
        if aas_estimation is None:
            raise NotImplementedError('AASNetwork estimation not provided. This case is not implemented yet !')

        return aas_estimation.get_action_prob(action, random_action_temperature)

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
                        param_idx = int(math.log2(param)) + 1 if param > 0 else 0
                        parallel_params_probs[i, param_idx] += child_p
                elif isinstance(child_action, Vectorization):
                    select_probs[Vectorization.ID] = child_p
                elif isinstance(child_action, NoTransformation):
                    select_probs[NoTransformation.ID] = child_p
                else:
                    raise ValueError(f'Action {child_action} is not supported !')
            # Correct parallelization parameters probabilities by calculating conditional probabilities
            print("Before correction")
            print(parallel_params_probs)
            parallel_prob_by_loop = parallel_params_probs.sum(dim=1)
            print("Parallel prob by loop")
            print(parallel_prob_by_loop)
            print("Parallel prob")
            print(parallel_prob)
            select_probs[Parallelization.ID] = parallel_prob
            for i in range(cfg.max_num_loops):
                if parallel_prob_by_loop[i] > 0:
                    # If the prior probability of select the loop is not zero, divide by that probability to get the conditional probability
                    parallel_params_probs[i] /= parallel_prob_by_loop[i]
                else:
                    # If the prior probability of select the loop is zero, set probability of not tiling to 1
                    parallel_params_probs[i, 0] = 1.0
            print("After correction")
            print(parallel_params_probs)
            print("Available actions")
            print(len(children_dict.keys()))
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
        # Disable gradients
        with torch.no_grad():
            # Set select probabilities
            select_probs = torch.zeros(cfg.num_transformations)
            select_probs[NoTransformation.ID] = 1.0
            # Set parallelization parameters probabilities
            parallel_params_probs = torch.zeros((cfg.max_num_loops, cfg.num_tile_sizes + 1))
            for i in range(cfg.max_num_loops):
                # Set probability of not tiling to 1
                parallel_params_probs[i, 0] = 1.0
            # Return the AASNetwork policy estimation
        return AASNetworkPolicyEstimation(
            select_probs=select_probs,
            parallel_params_probs=parallel_params_probs
        )
