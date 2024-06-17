import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.agent_utils import replay_buffer
from copy import deepcopy
import math
def to_numpy(var):
    return var.data.numpy()

################################################################################################################
# class QNetwork
#
# One hidden 2D conv with variable number of input channels.  We use 16 filters, a quarter of the original DQN
# paper of 64.  One hidden fully connected linear layer with a quarter of the original DQN paper of 512
# rectified units.  Finally, the output layer is a fully connected linear layer with a single output for each
# valid action.
#
################################################################################################################
class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(QNetwork, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=num_actions)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    # subclasses
    def forward(self, x):
        # Rectified output from the first conv layer
        x = F.relu(self.conv(x))

        # Rectified output from the final hidden layer
        x = F.relu(self.fc_hidden(x.view(x.size(0), -1)))

        # Returns the output from the fully-connected linear layer
        return self.output(x)

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            # if torch.cuda.is_available():
            #     param.data.copy_(torch.from_numpy(
            #         params[cpt:cpt + tmp]).view(param.size()).cuda())
            # else:
            param.data.copy_(torch.from_numpy(
                params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]


class Genome(object):
    def __init__(self, state_shape, num_actions, device,
                 gamma=0.99, learning_rate=3e-4, hard_replacement_interval=200, memory_size=1e5, batch_size=32,
                 init_epsilon=1.0, end_epsilon=0.1, epsilon_decay_steps=100000,):
        self.device = device
        self.in_channels = state_shape[2]
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.hr_interval = hard_replacement_interval
        self.batch_size = batch_size
        self.epsilon = init_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_delta = (init_epsilon - end_epsilon) / epsilon_decay_steps

        self.Q_net = QNetwork(self.in_channels, num_actions).to(self.device)
        self.Q_target = QNetwork(self.in_channels, num_actions).to(self.device)
        self.Q_target.load_state_dict(self.Q_net.state_dict())
        self.Q_net_optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=self.lr)

        self.replay_buffer = replay_buffer(buffer_size=int(memory_size))
        self.update_cnt = 0

    def epsilon_decay(self):
        self.epsilon = 0.1

    def select_action(self, state, is_greedy=False):
       
        state = (torch.tensor(state, device=self.device).permute(2, 0, 1)).unsqueeze(0).float()
        with torch.no_grad():
            action = self.Q_net(state).argmax(1).cpu().numpy()[0]
        return action
        
#        if (not is_greedy) and np.random.binomial(1, self.epsilon) == 1:
#            action = np.random.choice([i for i in range(self.num_actions)])
#        else:
#            state = (torch.tensor(state, device=self.device).permute(2, 0, 1)).unsqueeze(0).float()
#            with torch.no_grad():
#                action = self.Q_net(state).argmax(1).cpu().numpy()[0]
#        return action
        
        


class DQN(object):
    def __init__(self, state_shape, num_actions, device,
                 gamma=0.99, learning_rate=3e-4, hard_replacement_interval=200, memory_size=1e5, batch_size=32,
                 init_epsilon=1.0, end_epsilon=0.1, epsilon_decay_steps=100000,):
        self.device = device
        self.in_channels = state_shape[2]
        self.num_actions = num_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.hr_interval = hard_replacement_interval
        self.batch_size = batch_size
        self.epsilon = init_epsilon
        self.end_epsilon = end_epsilon
        self.epsilon_delta = (init_epsilon - end_epsilon) / epsilon_decay_steps

        self.Q_net = QNetwork(self.in_channels, num_actions).to(self.device)
        self.Q_target = QNetwork(self.in_channels, num_actions).to(self.device)
        self.Q_target.load_state_dict(self.Q_net.state_dict())
        self.Q_net_optimizer = torch.optim.Adam(self.Q_net.parameters(), lr=self.lr)
        self.best_fitness = -100000
        self.replay_buffer = replay_buffer(buffer_size=int(memory_size))
        self.update_cnt = 0

    def epsilon_decay(self):
        self.epsilon = max(self.epsilon - self.epsilon_delta, self.end_epsilon)

    def select_action(self, state, is_greedy=False):
        if (not is_greedy) and np.random.binomial(1, self.epsilon) == 1:
            action = np.random.choice([i for i in range(self.num_actions)])
        else:
            state = (torch.tensor(state, device=self.device).permute(2, 0, 1)).unsqueeze(0).float()
            with torch.no_grad():
                action = self.Q_net(state).argmax(1).cpu().numpy()[0]
        return action

    def store_experience(self, s, a, r, s_, done):
        self.replay_buffer.add(s, a, r, s_, done)

    def train(self,pop, best_Q_net, fitness):
        # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
        batch_samples = self.replay_buffer.sample(self.batch_size)

        # states, next_states are of tensor (BATCH_SIZE, in_channel, 10, 10) - inline with pytorch NCHW format
        # actions, rewards, is_terminal are of tensor (BATCH_SIZE, 1)
        states = torch.FloatTensor(np.stack(batch_samples.state, axis=0)).to(self.device).permute(0, 3, 1, 2).contiguous()
        actions = torch.LongTensor(np.stack(batch_samples.action, axis=0)).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(np.stack(batch_samples.reward, axis=0)).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(np.stack(batch_samples.next_state, axis=0)).to(self.device).permute(0, 3, 1, 2).contiguous()
        dones = torch.FloatTensor(np.stack(batch_samples.is_terminal, axis=0)).to(self.device).unsqueeze(1)

        # Obtain a batch of Q(S_t, A_t) and compute the forward pass.
        # Note: policy_network output Q-values for all the actions of a state, but all we need is the A_t taken at time t
        # in state S_t.  Thus we gather along the columns and get the Q-values corresponds to S_t, A_t.
        # Q_s_a is of size (BATCH_SIZE, 1).
        Q_sa = self.Q_net(states).gather(1, actions)

        # Obtain max_{a} Q(S_{t+1}, a) of any non-terminal state S_{t+1}.  If S_{t+1} is terminal, Q(S_{t+1}, A_{t+1}) = 0.
        # Note: each row of the network's output corresponds to the actions of S_{t+1}.  max(1)[0] gives the max action
        # values in each row (since this a batch).  The detach() detaches the target net's tensor from computation graph so
        # to prevent the computation of its gradient automatically.  Q_s_prime_a_prime is of size (BATCH_SIZE, 1).

        Q_sa_ = self.Q_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute the target
        target = rewards + self.gamma * (1 - dones) * Q_sa_

        # MSE loss
        loss = F.mse_loss(target, Q_sa)

        # Zero gradients, backprop, update the weights of policy_net
        self.Q_net_optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.Q_net.parameters(), 0.5)
        self.Q_net_optimizer.step()

        self.update_cnt += 1

        if self.update_cnt % self.hr_interval == 0:
            # Update the frozen target models
            self.Q_target.load_state_dict(self.Q_net.state_dict())
            # self.best_fitness = fitness
            # for ind in pop:
            #     ind.Q_target.load_state_dict(best_Q_net.state_dict())

            # remain for soft replacement
            # for param, target_param in zip(self.Q_net.parameters(), self.Q_target.parameters()):
            #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return loss.detach().cpu().numpy()

    def save(self, filename, directory):
        torch.save(self.Q_net.state_dict(), '%s/%s_q_net.pth' % (directory, filename))
        torch.save(self.Q_target.state_dict(), '%s/%s_q_target.pth' % (directory, filename))

    def load(self, filename, directory):
        self.Q_net.load_state_dict(torch.load('%s/%s_q_net.pth' % (directory, filename)))
        self.Q_target.load_state_dict(torch.load('%s/%s_q_target.pth' % (directory, filename)))
