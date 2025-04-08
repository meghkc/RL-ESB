# ppo_agent.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from config import STATE_DIM, ACTION_DIM, HIDDEN_SIZE, LEARNING_RATE, GAMMA, CLIP_EPS, GAE_LAMBDA, PPO_EPOCHS

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(ActorCritic, self).__init__()
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
        x = self.state_net(state)
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def act(self, state):
        state_tensor = torch.FloatTensor(state)
        logits, value = self.forward(state_tensor)
        # Action masking: use the continuous availability features from state[2:]
        avail = torch.FloatTensor(state[2:])  # shape: [ACTION_DIM]
        # Create mask: available if avail >= 0; else 0.
        mask = (avail >= 0).float().to(logits.device)
        very_negative = -1e9
        masked_logits = logits + (1 - mask) * very_negative
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_size):
        self.policy = ActorCritic(state_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)

    def compute_advantages(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + GAMMA * values[i+1] * (1 - dones[i]) - values[i]
            gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, trajectories):
        states = torch.FloatTensor(np.array([t[0] for t in trajectories]))
        actions = torch.LongTensor(np.array([t[1] for t in trajectories])).unsqueeze(1)
        old_log_probs = torch.stack([t[2] for t in trajectories]).detach()
        rewards = [t[3] for t in trajectories]
        dones = [1 if t[4] else 0 for t in trajectories]

        with torch.no_grad():
            _, state_values = self.policy(states)
        state_values = state_values.squeeze().detach().numpy().tolist()

        advantages = self.compute_advantages(rewards, state_values, dones)
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = advantages + torch.FloatTensor(state_values)

        for _ in range(PPO_EPOCHS):
            logits, values = self.policy(states)
            dist = Categorical(logits=logits)
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
