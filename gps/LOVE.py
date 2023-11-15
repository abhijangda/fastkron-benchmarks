#!/usr/bin/env python
# coding: utf-8

# # GP Regression with LOVE for Fast Predictive Variances and Sampling
# 
# ## Overview
# 
# In this notebook, we demonstrate that LOVE (the method for fast variances and sampling introduced in this paper https://arxiv.org/abs/1803.06058) can significantly reduce the cost of computing predictive distributions. This can be especially useful in settings like small-scale Bayesian optimization, where predictions need to be made at enormous numbers of candidate points.
# 
# In this notebook, we will train a KISS-GP model on the `skillcraft `UCI dataset, and then compare the time required to make predictions with each model.
# 
# **NOTE**: The timing results reported in the paper compare the time required to compute (co)variances __only__. Because excluding the mean computations from the timing results requires hacking the internals of GPyTorch, the timing results presented in this notebook include the time required to compute predictive means, which are not accelerated by LOVE. Nevertheless, as we will see, LOVE achieves impressive speed-ups.

# In[1]:


import math
import torch
import switch_KroneckerProduct
import gpytorch
from tqdm import tqdm

# from matplotlib import pyplot as plt

# Make plots inline
# get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading Data
# 
# For this example notebook, we'll be using the `elevators` UCI dataset used in the paper. Running the next cell downloads a copy of the dataset that has already been scaled and normalized appropriately. For this notebook, we'll simply be splitting the data using the first 40% of the data as training and the last 60% as testing.
# 
# **Note**: Running the next cell will attempt to download a small dataset file to the current directory.

# In[14]:


import urllib.request
import os
from scipy.io import loadmat
from math import floor

import sys

# this is for running the notebook in our testing framework
smoke_test = False #('CI' in os.environ)


# if not smoke_test and not os.path.isfile('../elevators.mat'):
#     print('Downloading \'elevators\' UCI dataset...')
#     urllib.request.urlretrieve('https://drive.google.com/uc?export=download&id=1jhWL3YUHvXIaftia4qeAyDwVxo6j1alk', '../elevators.mat')

# n = int(sys.argv[1])
# dims = int(sys.argv[2])
grid_size = int(sys.argv[1])
num_trace_samples = 10 #int(sys.argv[1])

if False:
    train_x = torch.zeros(n, dims)
    # for i in range(n):
    #     for j in range(dims):
    #         train_x[i * dims + j] = float(i) / (n-1)
    # True function is sin( 2*pi*(x0+x1))
    train_y = torch.sin((train_x[:, 0] + train_x[:, 1]) * (2 * math.pi)) + torch.randn_like(train_x[:, 0]).mul(0.01)
else:
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

# LOVE can be used with any type of GP model, including exact GPs, multitask models and scalable approximations. Here we demonstrate LOVE in conjunction with KISS-GP, which has the amazing property of producing **constant time variances.**
# 
# ## The KISS-GP + LOVE GP Model
# 
# We now define the GP model. For more details on the use of GP models, see our simpler examples. This model uses a `GridInterpolationKernel` (SKI) with an Deep RBF base kernel. The forward method passes the input data `x` through the neural network feature extractor defined above, scales the resulting features to be between 0 and 1, and then calls the kernel.
# 
# The Deep RBF kernel (DKL) uses a neural network as an initial feature extractor. In this case, we use a fully connected network with the architecture `d -> 1000 -> 500 -> 50 -> 2`, as described in the original DKL paper. All of the code below uses standard PyTorch implementations of neural network layers.

# In[3]:


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
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
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

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()


# ### Training the model
# 
# The cell below trains the GP model, finding optimal hyperparameters using Type-II MLE. We run 20 iterations of training using the `Adam` optimizer built in to PyTorch. With a decent GPU, this should only take a few seconds.

# In[5]:


training_iterations = 5 if smoke_test else 5


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

import time


def train():
    iterator = tqdm(range(training_iterations))
    for i in iterator:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        iterator.set_postfix(loss=loss.item())
        optimizer.step()
        
start = time.time()
with gpytorch.settings.use_toeplitz(False):
    with gpytorch.settings.num_trace_samples(num_trace_samples): # gpytorch.settings.fast_computations.log_prob(False):
        # with gpytorch.settings.max_preconditioner_size(0):
        #     with gpytorch.settings.min_preconditioning_size(0):
                with gpytorch.settings.max_cholesky_size(10):
                    with gpytorch.settings.debug(False):
                        train()
end = time.time()

print("Total training time ", (end - start)*1e3)