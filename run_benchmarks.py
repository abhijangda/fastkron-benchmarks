import os
import subprocess
import re
from functools import reduce
import time

class Shape:
  def __init__(self, m, n, p, q):
    self.m = m
    self.n = n
    self.ps = [p for i in range(0, n)]
    self.qs = [q for i in range(0, n)]
    self.k = reduce((lambda a, b: a * b), self.ps)

  def flops(self):
    ops = 0
    k = self.k
    for p,q in zip(reversed(self.ps),reversed(self.qs)):
      k = (k/p)*q
      ops += k * p
    return 2 * self.m * ops
    
  def __repr__(self):
    return f"{self.m}_{self.ps[0]}x{self.qs[0]}^{self.n}"

  def __eq__(self, other):
    return repr(self) == repr(other)

def run_command(command):
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print (f"Running {command}\n", o)
    assert False
  return o

class Executor:
  def __init__(self, exec_dir):
    self.exec_dir = exec_dir
    self.org_dir = None

  def __enter__(self):
    self.org_dir = os.getcwd()
    os.chdir(self.exec_dir)
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    os.chdir(self.org_dir)

class FastKronEval:
  def __init__(self, fk_dir):
    self.fk_dir = fk_dir
    self.last_shape = None

  def gen_kernels(self, shape, distKernels):
    run_command("python3 src/gen_tuner_kernels.py -distinct-factors " + \
                str(shape.n) + " " + " ".join([f"{pq[0]},{pq[1]}" for pq in zip(shape.ps, shape.qs)]) + \
                (" -dist-kernels " if distKernels else ""))

  def build_kron(self):
    run_command("make kron -j")

  def run_kron(self, shape, GM, GK, LocalKrons):
    with Executor(self.fk_dir) as executor:
      if GM*GK > 1:
        run_command("make gen-multi-gpu-tests-kernel")
      else: 
        self.gen_kernels(shape, False)
      self.build_kron()
      
      ld_path = "LD_LIBRARY_PATH="+self.fk_dir
      kron = ld_path + " " + f"./kron -m {shape.m} -n {shape.n} -p {shape.ps[0]} -q {shape.qs[0]} -r 20 -w 10 -t float --tune"
      if GM * GK != 1:
        kron += f" --gpus {GM*GK} --GM {GM} --GK {GK} --gpuLocalKrons 2"

      o = run_command(kron + " --fuse")
      fused = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
      fusedtime = re.findall(r"Time: ([\d\.]+) ms", o)[0]
      if shape.ps[0] <= 32:
        o = run_command(kron)
        wofuse = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
        wofusetime = re.findall(r"Time: ([\d\.]+) ms", o)[0]
      else:
        wofuse = fused
        wofusetime = fusedtime

      return (wofuse, wofusetime, fused, fusedtime)

  def run_distal(self, shape, GM, GK, LocalKrons):
    with Executor(self.fk_dir) as executor:
      if GM*GK > 1:
        run_command("make gen-multi-gpu-tests-kernel")
      else: 
        self.gen_kernels(shape, False)
      self.build_kron()
        
      ld_path = "LD_LIBRARY_PATH="+self.fk_dir + " DIST_COMM=NCCL"
      kron = ld_path + " " + f"./kron -m {shape.m} -n {shape.n} -p {shape.ps[0]} -q {shape.qs[0]} -r 20 -w 10 -t float --tune"
      if GM * GK != 1:
        kron += f" --gpus {GM*GK} --GM {GM} --GK {GK} --gpuLocalKrons 1"

      o = run_command(kron + " --fuse")
      fused = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
      fusedtime = re.findall(r"Time: ([\d\.]+) ms", o)[0]
      if shape.ps[0] <= 32:
        o = run_command(kron)
        wofuse = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
        wofusetime = re.findall(r"Time: ([\d\.]+) ms", o)[0]
      else:
        wofuse = fused
        wofusetime = fusedtime

      return (wofuse, wofusetime, fused, fusedtime)

class GPyTorchEval:
  def __init__(self):
    pass
  def run_kron(self, shape, GM, GK, LocalKrons):
    import gpytorch as gp
    import torch
    factors = []
    for p,q in zip(shape.ps, shape.qs):
        factors += [torch.ones(p, q, dtype=float).cuda()]    
    x = torch.ones(shape.m, shape.k, dtype=float).cuda()
    kp = gp.lazy.KroneckerProductLazyTensor(*factors)
    def run_case(r):
        t1 = time.time()
        for i in range(r):
            y = x @ kp
        torch.cuda.synchronize()
        t2 = time.time()
        return (t2-t1)*1000/r
    
    run_case(10)
    t = run_case(20)
    flops = shape.flops()/(t/1e3)
    torch.cuda.empty_cache()
    return (flops/1e9, t)

class CogentEval:
  def __init__(self, fk_bench_dir):
    self.fk_bench_dir = fk_bench_dir
  
  def run_kron(self, shape, GM, GK, LocalKrons):
    with Executor(os.path.join(self.fk_bench_dir, "TC-CGO2019/cogent/kron")) as exector:
      run_binary = f"run_{320 if shape.m == 1024 else shape.m}_{shape.n}_{shape.ps[0]}"
      run_command(f"make " + run_binary)
      o = run_command(f"./{run_binary} {shape.m} {shape.ps[0]}")
      try:
        t = re.findall(r'Time of one iteration: ([\d\.]+) milliseconds', o)[0]
        t = float(t)
        flops = shape.flops()/((t*shape.n)/1e3)
        return (flops/1e9, t*shape.n)
      except:
        print(f"./{run_binary} {shape.m} {shape.ps[0]}")
        print(o)
        return None

def run_single_gpu_large_M(fk_dir, fk_bench_dir):
  M = 1024
  M2 = 320
  cases = [Shape(M, 5, 8, 8),     Shape(M, 6, 8, 8),
           Shape(M, 4, 16, 16),   
           Shape(M2, 5, 16, 16),
           Shape(M, 3, 32, 32),   Shape(M2, 4, 32, 32),
           Shape(M, 2, 64, 64),   Shape(M, 3, 64, 64),
           Shape(M, 2, 128, 128), Shape(M2, 3, 128, 128)]
  fk_eval = FastKronEval(fk_dir)
  gpEval = GPyTorchEval()
  cogentEval = CogentEval(fk_bench_dir)

  for shape in cases:
    (wofuseflops, _, fuseflops, _) = fk_eval.run_kron(shape, 1, 1, 1)
    (gpflops, gptime) = gpEval.run_kron(shape, 1, 1, 1)
    (cogentflops, cogentime) = cogentEval.run_kron(shape, 1, 1, 1)
    print(shape, "&", fuseflops, "&", wofuseflops, "&", gpflops, '&', cogentflops)

def run_single_gpu_small_M(fk_dir, fk_bench_dir):
  M = 16
  cases = [Shape(M, 8, 8, 8),
           Shape(M, 6, 16, 16),
           Shape(M, 5, 32, 32),
           Shape(M, 4, 64, 64)]
  fk_eval = FastKronEval(fk_dir)
  gpEval = GPyTorchEval()
  cogentEval = CogentEval(fk_bench_dir)

  for shape in cases:
    (wofuseflops, _, fuseflops, _) = fk_eval.run_kron(shape, 1, 1, 1)
    (gpflops, gptime) = gpEval.run_kron(shape, 1, 1, 1)
    (cogentflops, cogentime) = cogentEval.run_kron(shape, 1, 1, 1)
    print(shape, "&", fuseflops, "&", wofuseflops, "&", gpflops, '&', cogentflops)

def run_multi_gpu(fk_dir):
  cases = []
  M_64 = 64
  cases += [Shape(M_64, 4, 64, 64)]
  M_128 = 4
  cases += [Shape(M_128, 4, 128, 128)]
  
  # run_command("make gen-multi-gpu-tests-kernel")
  fk_eval = FastKronEval(fk_dir)

  for shape in cases:
    GMs = [1, 2, 2, 4, 4]
    GKs = [1, 1, 2, 2, 4]
    for j,gpus in enumerate([1, 2, 4, 8, 16]):
      gm = GMs[j]
      gk = GKs[j]
      shapeGM = Shape(shape.m * gm, shape.n, shape.ps[0], shape.qs[0])
      (_, _, fkflops, _) = fk_eval.run_kron(shapeGM, gm, gk, 1)
      (_, _, distalflops, _) = fk_eval.run_distal(shapeGM, gm, gk, 1)
      print(shapeGM, "&", fkflops, "&", distalflops)

def do_evaluation(fk_dir, fk_bench):
  # run_single_gpu_large_M(fk_dir, fk_bench)
  # run_single_gpu_small_M(fk_dir, fk_bench)
  run_multi_gpu(fk_dir)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-fk-dir', required=True, type=str, help='Path to FastKron')
  parser.add_argument('-fk-bench-dir', required=True, type=str, help='Path to FastKron Benchmarks')
  args = parser.parse_args()

  try:
    import gpytorch
  except:
    print("GPyTorch is not installed")
    sys.exit(1)

  if not os.path.exists('/usr/local/cuda'):
    print("CUDA is not installed at /usr/local/cuda/")
    sys.exit(1)
  
  try:
    import torch
    if not torch.cuda.is_available():
      print("torch cuda is not available")
      sys.exit(1)
  except:
    print("torch is not installed")
    sys.exit(1)
  
  do_evaluation(os.path.abspath(args.fk_dir), os.path.abspath(args.fk_bench_dir))