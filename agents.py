from collections import defaultdict, namedtuple, deque
import numpy as np
import random
import math

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from models import QNetwork
from replay_buffers import PrioritizedReplayBuffer, ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PrioritizedDQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 buffer_size,
                 batch_size,
                 gamma,
                 tau,
                 lr,
                 lr_decay,
                 update_every,
                 update_mem_every,
                 update_mem_par_every,
                 experience_per_sampling,
                 seed,
                 epsilon,
                 epsilon_min,
                 eps_decay,
                 compute_weights,
                 hidden_1,
                 hidden_2,
                 ):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_decay = lr_decay
        self.update_every = update_every
        self.experience_per_sampling  = experience_per_sampling
        self.update_mem_every = update_mem_every
        self.update_mem_par_every = update_mem_par_every
        self.seed = random.seed(seed)
        self.epsilon= epsilon
        self.epsilon_min = epsilon_min
        self.eps_decay = eps_decay
        self.compute_weights = compute_weights
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.learn_steps = 0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.eps_decay = eps_decay
        self.compute_weights = compute_weights


        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_1, hidden_2).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_1, hidden_2).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=self.lr_decay)


        # Replay memory
        self.memory = PrioritizedReplayBuffer(
                    self.action_size,
                    self.buffer_size,
                    self.batch_size,
                    self.experience_per_sampling,
                    self.seed,
                    self.compute_weights)
        # Initialize time step (for updating every UPDATE_NN_EVERY steps)
        self.t_step_nn = 0
        # Initialize time step (for updating every UPDATE_MEM_PAR_EVERY steps)
        self.t_step_mem_par = 0
        # Initialize time step (for updating every UPDATE_MEM_EVERY steps)
        self.t_step_mem = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_NN_EVERY time steps.
        self.t_step_nn = (self.t_step_nn + 1) % self.update_every
        self.t_step_mem = (self.t_step_mem + 1) % self.update_mem_every
        self.t_step_mem_par = (self.t_step_mem_par + 1) % self.update_mem_par_every
        if self.t_step_mem_par == 0:
            self.memory.update_parameters()
        if self.t_step_nn == 0:
            # If enough samples are available in memory, get random subset and learn
            if self.memory.experience_count > self.experience_per_sampling:
                sampling = self.memory.sample()
                self.learn(sampling)
        if self.t_step_mem == 0:
            self.memory.update_memory_sampling()

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
        """
        self.epsilon = max(self.epsilon*self.eps_decay, self.epsilon_min)

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            #print(action_values)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            #print(np.argmax(action_values.cpu().data.numpy()))
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, sampling):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            sampling (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, weights, indices  = sampling

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        if self.compute_weights:
            with torch.no_grad():
                weight = sum(np.multiply(weights, loss.data.cpu().numpy()))
            loss *= weight

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.learn_steps += 1
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        # ------------------- update priorities ------------------- #
        delta = abs(Q_targets - Q_expected.detach()).cpu().numpy()
        self.memory.update_priorities(delta, indices)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class DDQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 buffer_size,
                 batch_size,
                 gamma,
                 tau,
                 lr,
                 hidden_1,
                 hidden_2,
                 update_every,
                 epsilon,
                 epsilon_min,
                 eps_decay,
                 seed
                 ):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.seed = random.seed(seed)
        self.learn_steps = 0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.eps_decay = eps_decay

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_1, hidden_2).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_1, hidden_2).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def step(self, state, action, reward, next_state,  done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state, A_next=False):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
        """
        self.epsilon = max(self.epsilon*self.eps_decay, self.epsilon_min)
        if A_next:
            state = torch.from_numpy(state).float().to(device)
    
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            if A_next:
                return np.argmax(action_values.cpu().data.numpy(), axis = 1).reshape(-1, 1)
            else:
                return np.argmax(action_values.cpu().data.numpy())
        else:
            if A_next:
                return np.random.choice(np.arange(self.action_size), size=(self.batch_size, 1))
            else:
                return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
     
        A_next = self.act(next_states.detach().cpu().numpy().reshape(self.batch_size, -1), A_next=True)
        Q_next = self.qnetwork_target(next_states).gather(1, torch.from_numpy(np.asarray(A_next)).to(device))
        Q_targets = rewards + (self.gamma * Q_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_steps += 1

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
            
            
class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self,
                 state_size,
                 action_size,
                 buffer_size,
                 batch_size,
                 gamma,
                 tau,
                 lr,
                 hidden_1,
                 hidden_2,
                 update_every,
                 epsilon,
                 epsilon_min,
                 eps_decay,
                 seed
                 ):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every
        self.seed = random.seed(seed)
        self.learn_steps = 0
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.eps_decay = eps_decay

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, hidden_1, hidden_2).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, hidden_1, hidden_2).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0


    def step(self, state, action, reward, next_state,  done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # Sample if enough samples are available
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)

    def act(self, state):
        """Returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
        """
        self.epsilon = max(self.epsilon*self.eps_decay, self.epsilon_min)

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.learn_steps += 1

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

