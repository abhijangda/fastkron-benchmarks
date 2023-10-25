import os
import subprocess
import re

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

def getstatusoutput(command):
    (s, o) = subprocess.getstatusoutput(command)
    if s != 0:
        print("error executing ", command)
        print("output", o)
    return o

M = 1024
cases = [
    # Shape(M, 5, 8, 8),     Shape(M, 6, 8, 8),
       Shape(M, 4, 16, 16),   Shape(M//2, 5, 16, 16),
        Shape(M, 3, 32, 32),   Shape(M//2, 4, 32, 32),
        Shape(M, 2, 64, 64),   Shape(M//2, 3, 64, 64),
        Shape(M, 2, 128, 128), Shape(M//4, 3, 128, 128)]

cwd = os.getcwd()
os.chdir("TC-CGO2019/cogent/kron")

for case in cases:
    # print(getstatusoutput("ls -a"))
    getstatusoutput(f"make run_320_{case.n}_{case.ps[0]}")
    o = getstatusoutput(f"./run_320_{case.n}_{case.ps[0]} {case.m} {case.ps[0]}")
    t = re.findall(r'Time of one iteration: ([\d\.]+) milliseconds', o)[0]
    t = float(t)
    flops = case.flops()/((t*case.n)/1e3)
    print(case, "GFLOPS", flops/1e9)