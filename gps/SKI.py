import os, sys
import gc
import copy
import math
import torch
import time
from scipy.io import loadmat

# import set_managed_allocator

# from matplotlib import pyplot as plt
import numpy as np

from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm
#from sgkigp.gpyexample.gridkernel import CustomGridInterpolationKernel

import switch_KroneckerProduct
import gpytorch

from gpytorch.kernels import GridInterpolationKernel

# Source: https://docs.gpytorch.ai/en/latest/examples/02_Scalable_Exact_GPs/KISSGP_Regression.html

import traceback
import sys

def format_exception(e):
    exception_list = traceback.format_stack()
    exception_list = exception_list[:-2]
    exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
    exception_list.extend(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))

    exception_str = "Traceback (most recent call last):\n"
    exception_str += "".join(exception_list)
    # Removing the last \n
    exception_str = exception_str[:-1]

    return exception_str

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

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
            super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)

            # SKI requires a grid size hyperparameter. This util can help with that

            self.mean_module = gpytorch.means.ConstantMean()
            # self.covar_module = gpytorch.kernels.ScaleKernel(
            #     GridInterpolationKernel(
            #         gpytorch.kernels.RBFKernel(), grid_size=grid_size, num_dims=dims
            #     )
            # )
            
            # self.covar_module = gpytorch.kernels.ProductStructureKernel(gpytorch.kernels.ScaleKernel(GridInterpolationKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dims),
            #                                             grid_size=grid_size, num_dims=1)), num_dims=dims)

            self.covar_module = gpytorch.kernels.ScaleKernel(GridInterpolationKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dims),
                                                        grid_size=grid_size, num_dims=dims))
            # if True:
            #     self.covar_module = gpytorch.kernels.ProductStructureKernel(
            #         gpytorch.kernels.ScaleKernel(GridInterpolationKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dims),
            #                                             grid_size=grid_size, num_dims=1)),
            #         num_dims = dims
            #     )

    def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    # model = train_to_covergence(model, train_x, train_y)

def train(training_iterations=5):
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPRegressionModel(train_x, train_y, likelihood)

    # Find optimal model hyperparameters
    model.cuda()
    likelihood.cuda()

    model.train()
    likelihood.train()

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    torch.cuda.synchronize()
    import time
    print('Tuning hyper-parameters ...')
    start = time.time()
    for i in range(training_iterations):
        print ("iteration ", i)
        # optimizer.zero_grad()
        output = model(train_x)
        # output@torch.ones(dims**grid_size, 11)
        # try:
        loss = -mll(output, train_y)
        loss.backward()
        # except torch.cuda.OutOfMemoryError as e:
        #     import gc
        #     print(format_exception(e))
        #     for obj in gc.get_objects():
        #         try:
        #             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
        #                 print(type(obj), obj.size(), obj.numel() * obj.element_size())
        #         except:
        #             pass
        #     return
        optimizer.step()
    torch.cuda.synchronize()
    end = time.time()
    print("Total time ", (end - start)*1e3, " ms")
    print("KronMatmulTime ", KronMatmulTime, " ms")
    # from torch.profiler import *
    # with profile(activities=[ProfilerActivity.CUDA],
    #     profile_memory=True, record_shapes=True) as prof:
    #     try:
    #         train()
    #     except:
    #         pass

    # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

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
                            with  gpytorch.settings.max_cg_iterations(20):
                                # with  gpytorch.settings.terminate_cg_by_size(0):
                                    with gpytorch.settings.verbose_linalg(True):
                                        train()