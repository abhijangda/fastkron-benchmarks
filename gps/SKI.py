import os, sys
import gc
import copy
import math
import torch
import time
from scipy.io import loadmat
import numpy as np
import gpytorch
import sys

KronMatmulTime = 0

class SKI(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood, dims, grid_size):
    super(SKI, self).__init__(train_x, train_y, likelihood)
    self.mean_module = gpytorch.means.ConstantMean()
    self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.GridInterpolationKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dims),
                                                    grid_size=grid_size, num_dims=dims))

  def forward(self, x):
    mean_x = self.mean_module(x)
    covar_x = self.covar_module(x)
    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class LOVE(gpytorch.models.ExactGP):
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

  def __init__(self, train_x, train_y, likelihood, dims, grid_size):
    super(LOVE, self).__init__(train_x, train_y, likelihood)
    
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

def train(klass, dataset, dataset_name, grid_size):
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

  likelihood = gpytorch.likelihoods.GaussianLikelihood()
  model = klass(train_x, train_y, likelihood, grid_size, dims)

  # Find optimal model hyperparameters
  model.cuda()
  likelihood.cuda()

  model.train()
  likelihood.train()

  mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

  # Use the adam optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
  torch.cuda.synchronize()
  start = time.time()
  for i in range(training_iterations):
    print ("i = ", i)
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()
  
  torch.cuda.synchronize()
  end = time.time()
  print("Total time ", (end - start)*1e3, " ms")
  print("KronMatmulTime ", KronMatmulTime, " ms")

gpytorch.settings.max_root_decomposition_size(0)  
gpytorch.settings.use_toeplitz(False)
gpytorch.settings.fast_computations.log_prob(True)
gpytorch.settings.debug(False)
gpytorch.settings.cg_tolerance(0)
gpytorch.settings.max_cholesky_size(0)
gpytorch.settings.max_cg_iterations(20)
gpytorch.settings.verbose_linalg(True)

import linear_operator
import linear_operator.operators.kronecker_product_linear_operator
orig_matmul = linear_operator.operators.kronecker_product_linear_operator._matmul
orig_t_matmul = linear_operator.operators.kronecker_product_linear_operator._t_matmul

def switch_KroneckerProduct(use_fastkron):
  global KronMatmulTime
  
  KronMatmulTime = 0
  def new_matmul(linear_ops, kp_shape, rhs):
      global KronMatmulTime
      torch.cuda.synchronize()
      s = time.time()
      if use_fastkron:
        res = rhs
      else:
        res = orig_matmul(linear_ops, kp_shape, rhs)
      torch.cuda.synchronize()
      e = time.time()
      KronMatmulTime += (e - s) * 1000
      return res

  def new_t_matmul(linear_ops, kp_shape, rhs):
      global KronMatmulTime
      torch.cuda.synchronize()
      s = time.time()
      if use_fastkron:
        res = rhs
      else:
        res = orig_t_matmul(linear_ops, kp_shape, rhs)
      torch.cuda.synchronize()
      e = time.time()
      KronMatmulTime += (e - s) * 1000
      return res

  linear_operator.operators.kronecker_product_linear_operator._matmul = new_matmul
  linear_operator.operators.kronecker_product_linear_operator._t_matmul = new_t_matmul

if __name__ == "__main__":
  gptype = sys.argv[1]
  grid_size = int(sys.argv[2])
  num_trace_samples = int(sys.argv[3])
  dataset = sys.argv[4]
  dataset_name = sys.argv[5]

  with gpytorch.settings.num_trace_samples(num_trace_samples):
    train(SKI, dataset, dataset_name, grid_size)
  