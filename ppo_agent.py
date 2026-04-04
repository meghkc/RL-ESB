# ppo_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from config import HIDDEN_SIZE, LEARNING_RATE, GAMMA, CLIP_EPS, GAE_LAMBDA, PPO_EPOCHS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for PPO.

    This class defines a neural network with a shared body and two heads:
    one for the actor (policy) and one for the critic (value function).

    Attributes:
        state_net (nn.Sequential): The shared layers of the network.
        actor (nn.Linear): The actor-specific layer.
        critic (nn.Linear): The critic-specific layer.
    """
    def __init__(self, state_dim, action_dim, hidden_size):
        """
        Initializes the ActorCritic network.

        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space.
            hidden_size (int): The number of neurons in the hidden layers.
        """
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 32),
            nn.ReLU()
        )
        self.actor = nn.Linear(32, action_dim)
        self.critic = nn.Linear(32, 1)

    def forward(self, state):
        """
        Performs the forward pass through the network.

        Args:
            state (torch.Tensor): The input state.

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The action logits.
                - torch.Tensor: The state value.
        """
        x = self.state_net(state)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    @staticmethod
    def apply_action_mask(logits, action_mask):
        if action_mask is None:
            return logits
        very_negative = -1e9
        return logits.masked_fill(~action_mask.bool(), very_negative)

    def build_action_mask(self, state, valid_actions=None):
        mask = torch.zeros(self.action_dim, dtype=torch.bool, device=device)

        if valid_actions is not None:
            valid_indices = [idx for idx in valid_actions if 0 <= idx < self.action_dim]
            if valid_indices:
                mask[valid_indices] = True
            else:
                mask[:] = True
            return mask

        # Backward-compatible fallback: infer availability from state[2:].
        availability = np.asarray(state[2 : 2 + self.action_dim], dtype=np.float32)
        if availability.shape[0] == self.action_dim:
            mask = torch.from_numpy(availability >= 0).to(device)
            if not torch.any(mask):
                mask[:] = True
            return mask

        mask[:] = True
        return mask

    def act(self, state, valid_actions=None):
        """
        Selects an action based on the current policy.

        Args:
            state (np.ndarray): The current state.
            valid_actions (list, optional): Valid action ids for this state.

        Returns:
            tuple: A tuple containing:
                - int: The selected action.
                - torch.Tensor: The log probability of the selected action.
                - torch.Tensor: The estimated state value.
        """
        state_tensor = torch.FloatTensor(state).to(device)
        logits, value = self.forward(state_tensor)
        action_mask = self.build_action_mask(state, valid_actions=valid_actions)
        masked_logits = self.apply_action_mask(logits, action_mask)
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.

    This class implements the PPO algorithm, which is an on-policy
    actor-critic method that uses a clipped surrogate objective function
    to improve training stability.

    Attributes:
        policy (ActorCritic): The actor-critic network.
        optimizer (torch.optim.Optimizer): The optimizer for the network.
    """
    def __init__(self, state_dim, action_dim, hidden_size):
        """
        Initializes the PPOAgent.

        Args:
            state_dim (int): The dimension of the state space.
            action_dim (int): The dimension of the action space.
            hidden_size (int): The number of neurons in the hidden layers.
        """
        self.policy = ActorCritic(state_dim, action_dim, hidden_size).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.action_dim = action_dim

    def compute_advantages(self, rewards, values, dones):
        """
        Computes the advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards (list): A list of rewards.
            values (list): A list of state values.
            dones (list): A list of done flags.

        Returns:
            list: A list of advantages.
        """
        advantages = []
        gae = 0
        values = values + [0]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, trajectories):
        """
        Updates the actor-critic network using the PPO algorithm.

        Args:
            trajectories (list): A list of trajectories. Each item is either:
                (state, action, log_prob, reward, done) or
                (state, action, log_prob, reward, done, valid_actions).
        """
        states = torch.FloatTensor(np.array([t[0] for t in trajectories])).to(device)
        actions = torch.LongTensor(np.array([t[1] for t in trajectories])).unsqueeze(1).to(device)
        old_log_probs = torch.stack([t[2] for t in trajectories]).detach().to(device)
        rewards = [t[3] for t in trajectories]
        dones = [1 if t[4] else 0 for t in trajectories]
        valid_actions_batch = [t[5] if len(t) > 5 else None for t in trajectories]

        action_masks = torch.stack(
            [
                self.policy.build_action_mask(t[0], valid_actions=valid_actions_batch[idx])
                for idx, t in enumerate(trajectories)
            ]
        ).to(device)

        with torch.no_grad():
            _, state_values = self.policy(states)
        state_values = state_values.squeeze().detach().cpu().numpy().tolist()

        advantages = self.compute_advantages(rewards, state_values, dones)
        advantages = torch.FloatTensor(advantages).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + torch.FloatTensor(state_values).to(device)

        for _ in range(PPO_EPOCHS):
            logits, values = self.policy(states)
            masked_logits = self.policy.apply_action_mask(logits, action_masks)
            dist = Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(actions.squeeze())

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
