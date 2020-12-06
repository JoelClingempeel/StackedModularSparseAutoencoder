import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# For a batch of vectors.
def k_mask(x, k):
    thresholds = x.topk(k).values[:, -1]
    return x >= (thresholds.unsqueeze(1) * torch.ones(x.shape))

# For a batch of striped vectors.
def stripe_k_mask(x, k):
    return k_mask(torch.mean(x, 2), k) > torch.zeros(x.shape[:-1])

class Net(nn.Module):
    def __init__(self,
                 input_dim,
                 stripe_dim,
                 num_stripes,
                 num_active_stripes,
                 distort_prob,
                 device):
        super(Net, self).__init__()
        self.layer_enc = nn.Linear(input_dim, stripe_dim * num_stripes)
        self.layer_dec = nn.Linear(stripe_dim * num_stripes, input_dim)

        self.stripe_dim = stripe_dim
        self.num_stripes = num_stripes
        self.num_active_stripes = num_active_stripes

        self.distort_prob = distort_prob
        self.routing_layer = nn.Linear(input_dim, num_stripes)
        self.device = device

    def _distort_mask(self, mask):
        rand_mask = torch.rand(mask.shape) > 1 - self.distort_prob
        return torch.logical_xor(mask, rand_mask)

    def routing_sparsify_stripes(self, input_data, stripe_data):
        routing_scores = self.routing_layer(input_data)
        mask = self._distort_mask(k_mask(routing_scores, self.num_active_stripes))
        return mask.unsqueeze(2) * stripe_data

    def encode(self, x):
        stripe_data = F.relu(self.layer_enc(x))
        stripe_data = stripe_data.reshape(-1, self.num_stripes, self.stripe_dim)
        return self.routing_sparsify_stripes(x, stripe_data)

    def decode(self, x):
        x = x.reshape(-1, self.num_stripes * self.stripe_dim)
        return F.relu(self.layer_dec(x))

    def forward(self, x):
        return self.decode(self.encode(x))

    def get_active_stripes(self, x):
        code = self.encode(x).squeeze(0)
        zero_stripe = torch.zeros(self.stripe_dim).to(self.device)
        return [j for j, stripe in enumerate(code)
                if not torch.all(torch.eq(stripe, zero_stripe))]

    def get_stripe_stats(self, X, Y):
        activations = {}
        for i in range(10):
            activations[i] = {}
            for j in range(self.num_stripes):
                activations[i][j] = 0

        for k in range(len(Y)):
            digit = Y[k]
            x_var = torch.FloatTensor(X[k: k + 1])
            x_var = x_var.to(self.device) 
            stripes = self.get_active_stripes(x_var)
            for stripe in stripes:
                activations[digit][stripe] += 1
        return activations

    def get_average_activations(self, X, Y):
        running_activations = {}
        running_counts = {}
        for digit in range(10):
            running_activations[str(digit)] = torch.zeros(self.num_stripes, self.stripe_dim).to(self.device)
            running_counts[str(digit)] = 0

        with torch.no_grad():
            for datum, label in zip(X, Y):
                x_var = torch.FloatTensor(datum).unsqueeze(0)
                x_var = x_var.to(device)
                digit = str(label.item())
                running_activations[digit] += self.encode(x_var).squeeze(0)
                running_counts[digit] += 1

        return torch.stack([running_activations[str(digit)] / running_counts[str(digit)]
                            for digit in range(10)],
                           dim=0).to(self.device)
