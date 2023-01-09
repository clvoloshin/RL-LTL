import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy(nn.Module):
    def __init__(self, obs_space, act_space):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(obs_space, 16)
        # self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(16, act_space)
        self.affine2_ = nn.Linear(16, act_space)

        self.affine3 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.affine1(x)
        # x = self.dropout(x)
        x = F.relu(x)
        mus = self.affine2(x)
        sigma_sqs = self.affine2_(x)

        rho = self.affine3(x)

        return (mus, sigma_sqs), rho