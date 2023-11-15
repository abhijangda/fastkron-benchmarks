#!/usr/bin/env python
# coding: utf-8

import math
import switch_KroneckerProduct
import gpytorch
import torch
import time
from tqdm import tqdm

import urllib.request
import os
from scipy.io import loadmat
from math import floor

import sys

# n = int(sys.argv[1])
# dims = int(sys.argv[2])
grid_size = int(sys.argv[1])
num_trace_samples = 100 #int(sys.argv[1])

dataset = sys.argv[2]
dataset_name = sys.argv[3]
data = torch.Tensor(loadmat(os.path.join(dataset,dataset_name+".mat"))['data'])
X = data[:, :-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:, -1]
train_n = int(math.floor(0.64*len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()
dims = train_x.shape[1]
test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

print("N =", dims, "; P =", grid_size)
train_x = train_x.cuda()
train_y = train_y.cuda()

class LargeFeatureExtractor(torch.nn.Sequential):           
    def __init__(self, input_dim):                                      
        super(LargeFeatureExtractor, self).__init__()        
        self.add_module('linear1', torch.nn.Linear(input_dim, 1000))
        self.add_module('relu1', torch.nn.ReLU())                  
        self.add_module('linear2', torch.nn.Linear(1000, 500))     
        self.add_module('relu2', torch.nn.ReLU())                  
        self.add_module('linear3', torch.nn.Linear(500, 50))       
        self.add_module('relu3', torch.nn.ReLU())                  
        self.add_module('linear4', torch.nn.Linear(50, dims))         


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.GridInterpolationKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dims)),
            grid_size=grid_size, num_dims=dims,
        )
        
        # Also add the deep net
        self.feature_extractor = LargeFeatureExtractor(input_dim=train_x.size(-1))

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        # We're also scaling the features so that they're nice values
        projected_x = self.feature_extractor(x)
        projected_x = projected_x - projected_x.min(0)[0]
        projected_x = 2 * (projected_x / projected_x.max(0)[0]) - 1
        
        # The rest of this looks like what we've seen
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPRegressionModel(train_x, train_y, likelihood)

model = model.cuda()
likelihood = likelihood.cuda()


# ### Training the model
# 
# The cell below trains the GP model, finding optimal hyperparameters using Type-II MLE. We run 20 iterations of training using the `Adam` optimizer built in to PyTorch. With a decent GPU, this should only take a few seconds.

# In[5]:


training_iterations = 5


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

import time


def train():
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        
start = time.time()
with gpytorch.settings.max_root_decomposition_size(0):
    with gpytorch.settings.use_toeplitz(False):
    # with  gpytorch.settings.fast_computations(, True, True):
        with gpytorch.settings.num_trace_samples(num_trace_samples): 
            with gpytorch.settings.fast_computations.log_prob(True):
    # with gpytorch.settings.max_preconditioner_size(0):
    # with gpytorch.settings.min_preconditioning_size(10):
                with gpytorch.settings.debug(False):
                    with gpytorch.settings.cg_tolerance(0):
                        with gpytorch.settings.max_cholesky_size(0):
                            with  gpytorch.settings.max_cg_iterations(100):
                                # with  gpytorch.settings.terminate_cg_by_size(0):
                                    with gpytorch.settings.verbose_linalg(True):
                                        train()
end = time.time()

print("Total training time ", (end - start)*1e3)
print("Kronecker time ", switch_KroneckerProduct.KronMatmulTime)