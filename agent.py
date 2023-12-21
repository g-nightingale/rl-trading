import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
    
class Agent(nn.Module):
    def __init__(self, n_actions, n_channels, grid_width, grid_height, device='cpu'):
        super(Agent, self).__init__()
        self.n_actions = n_actions
        self.n_channels = n_channels
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.unrolled_conv_size = self.get_unrolled_conv_size()
        self.conv1 = nn.Conv2d(n_channels, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self.unrolled_conv_size, 128)
        self.action_head = nn.Linear(128, n_actions)
        self.value_head = nn.Linear(128, 1)
        self.device = device

    def get_unrolled_conv_size(self):
        # Calculate dimensions after first convolution
        conv1_width = self.grid_width - 3 + 1
        conv1_height = self.grid_height - 3 + 1
        
        # Calculate dimensions after second convolution
        conv2_width = conv1_width - 3 + 1
        conv2_height = conv1_height - 3 + 1
        
        # Return total number of dimensions for unrolled conv2 layer
        return 32 * conv2_width * conv2_height
    
    def forward(self, x):
        x = torch.tanh(self.conv1(x))        
        x = torch.tanh(self.conv2(x))        
        x = x.view(-1, self.unrolled_conv_size)
        x = torch.tanh(self.fc1(x))
        return self.value_head(x), self.action_head(x)

    def get_action(self, x):       
        _, logits = self(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action.item()
    
    def get_value(self, x):
        return self(x)[0]

    def get_action_and_value(self, x, action=None):
        value, logits = self(x)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value