import gpytorch as gp
import torch
import time

class Shape:
  def __init__(self, m, n, p, q):
    self.m = m
    self.n = n
    self.ps = [p for i in range(0, n)]
    self.qs = [q for i in range(0, n)]
    self.k = 1
    for p in self.ps:
        self.k *= p

  def __repr__(self):
    return f"{self.m}_{self.ps[0]}x{self.qs[0]}^{self.n}"

  def flops(self):
    ops = 0
    k = self.k
    for p,q in zip(reversed(self.ps),reversed(self.qs)):
      k = (k/p)*q
      ops += k * p
    return 2 * self.m * ops

warmup = 10
epochs = 20

def run_cases(cases):
  for case in cases:
    factors = []
    for p,q in zip(case.ps, case.qs):
        factors += [torch.ones(p,q, dtype=float).cuda()]    
    x = torch.ones(case.m, case.k, dtype=float).cuda()
    kp = gp.lazy.KroneckerProductLazyTensor(*factors)
    def run_case(r):
        t1 = time.time()
        for i in range(r):
            y = x @ kp
        torch.cuda.synchronize()
        t2 = time.time()
        return (t2-t1)*1000/r
    
    run_case(1)
    run_case(epochs)
    t = run_case(warmup)
    flops = case.flops()/(t/1e3)
    print(case, "%.2f"%(flops/1e9), "%.2f"%t)

M = 1024
big_cases = [Shape(M, 5, 8, 8),     Shape(M, 6, 8, 8),
        Shape(M, 4, 16, 16),   Shape(M//2, 5, 16, 16),
        Shape(M, 3, 32, 32),   Shape(M//2, 4, 32, 32),
        Shape(M, 2, 64, 64),   Shape(M//2, 3, 64, 64),
        Shape(M, 2, 128, 128), Shape(M//4, 3, 128, 128)]

# run_cases(big_cases)

M = 16
small_cases = [Shape(M, 8, 8, 8),
          Shape(M, 6, 16, 16),
          Shape(M, 5, 32, 32),
          Shape(M, 4, 64, 64),
        #  Shape(M, 3, 128, 128)
          ]
run_cases(small_cases)