import os
import subprocess
import re
from functools import reduce
import time
import math
import shutil

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
  def __str__(self):
    return repr(self)
  def __eq__(self, other):
    return repr(self) == repr(other)

def run_command(command):
  print("Running ", command, " in directory ", os.getcwd())
  (s, o) = subprocess.getstatusoutput(command)
  if s != 0:
    print (f"Error running {command}\n", o)
    assert False
  return o

def total_gpu_memory():
  o = run_command("nvidia-smi -d MEMORY -q -i 0")
  mems = re.findall(r'Total\s*\:\s*(\d+)', o)
  mems = [int(m) for m in mems]
  return max(mems)

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

  def setup_cmake(self):
    if os.path.exists('build/'):
      shutil.rmtree('build/')
    os.mkdir('build/')
    os.chdir('build/')
    run_command('cmake ..')

  def build_kron(self):
    run_command("make benchmark -j")

  def run_kron(self, shape, GM, GK, LocalKrons, callnofuse=True):
    with Executor(self.fk_dir) as executor: 
      self.gen_kernels(shape, False)
      self.setup_cmake()
      if GM*GK > 1:
        run_command("make gen-multi-gpu-tests-kernel")
      self.build_kron()
      
      ld_path = "LD_LIBRARY_PATH="+self.fk_dir
      kron = ld_path + " " + f"./benchmark -m {shape.m} -n {shape.n} -p {shape.ps[0]} -q {shape.qs[0]} -r 20 -w 10 -t float --tune"
      if GM * GK != 1:
        kron += f" --gpus {GM*GK} --GM {GM} --GK {GK} --gpuLocalKrons 2"

      o = run_command(kron + " --fuse")
      fused = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
      fusedtime = re.findall(r"Time: ([\d\.]+) ms", o)[0]
      if shape.ps[0] <= 32 and callnofuse:
        o = run_command(kron)
        wofuse = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
        wofusetime = re.findall(r"Time: ([\d\.]+) ms", o)[0]
      else:
        wofuse = fused
        wofusetime = fusedtime

      return (wofuse, wofusetime, fused, fusedtime)

  def run_distal(self, shape, GM, GK, LocalKrons):
    with Executor(self.fk_dir) as executor:
      self.gen_kernels(shape, False)
      self.setup_cmake()
      if GM*GK > 1:
        run_command("make gen-multi-gpu-tests-kernel")
      self.build_kron()
        
      ld_path = "LD_LIBRARY_PATH="+self.fk_dir + " DIST_COMM=NCCL"
      kron = ld_path + " " + f"./benchmark -m {shape.m} -n {shape.n} -p {shape.ps[0]} -q {shape.qs[0]} -r 20 -w 10 -t float --tune"
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
    r = self._run_kron(shape, GM, GK, LocalKrons)
    torch.cuda.empty_cache()
    return r

  def _run_kron(self, shape, GM, GK, LocalKrons):
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
    return (flops/1e9, t)

class CogentEval:
  def __init__(self, fk_bench_dir):
    self.fk_bench_dir = fk_bench_dir
  
  def run_kron(self, shape):
    with Executor(os.path.join(self.fk_bench_dir, "TC-CGO2019/cogent/kron")) as exector:
      o = run_command(f"python3 gen_kernels.py {shape.m} {shape.n} {shape.ps[0]}")
      o = run_command(f'nvcc -O3 -std=c++11 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70 mains/main_{shape.n}_facs.c kernels/kernel_{shape.m}_{shape.n}_{shape.ps[0]}.cu -Xptxas "-v " -o run_temp')
      o = run_command(f"./run_temp {shape.m} {shape.ps[0]}")
      return self.parse_output(shape, o)

  def parse_output(self, shape, output):
    try:
      t = re.findall(r'Time of one iteration: ([\d\.]+) milliseconds', output)[0]
      t = float(t)
      flops = shape.flops()/((t*shape.n)/1e3)
      return (flops/1e9, t*shape.n)
    except:
      print(f"./{run_binary} {shape.m} {shape.ps[0]}")
      print(o)
      return None

def run_single_gpu_large_M(fk_dir, fk_bench_dir):
  resultsCSV = ""
  M = 1024
  M2 = 320
  if total_gpu_memory() <= 16*1024:
    M = M/2
    M2 = M2/2
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
    (cogentflops, cogentime) = cogentEval.run_kron(shape)
    result = f"{str(shape)} & {fuseflops} & {wofuseflops} & {gpflops} & {cogentflops}"
    print("TFLOPs", result)
    resultsCSV += result + "\n"

  with open(os.path.join(fk_bench_dir, "single-gpu-flops.csv"), "w") as f:
    f.write(resultsCSV)

def run_single_gpu_small_M(fk_dir, fk_bench_dir):
  M = 16
  if total_gpu_memory() <= 16*1024:
    M = M/2
  cases = [Shape(M, 8, 8, 8),
           Shape(M, 6, 16, 16),
           Shape(M, 5, 32, 32),
           Shape(M, 4, 64, 64)]
  fk_eval = FastKronEval(fk_dir)
  gpEval = GPyTorchEval()
  cogentEval = CogentEval(fk_bench_dir)
  
  floatResultsCSV = "P & N & FastKron & COGENT & GPyTorch\n"
  doubleResultsCSV = "P & N & FastKron & COGENT & GPyTorch\n"
  for shape in cases:
    (wofuseflops, _, fuseflops, _) = fk_eval.run_kron(shape, 1, 1, 1)
    (gpflops, gptime) = gpEval.run_kron(shape, 1, 1, 1)
    (cogentflops, cogentime) = cogentEval.run_kron(shape)
    floatResultsCSV += f"{shape.ps[0]} & {shape.n} & {fuseflops} & {cogentflops} & {gpflops}" + "\n"
    doubleResultsCSV += f"{shape.ps[0]} & {shape.n} & {float(fuseflops)/2} & {float(cogentflops)/2} & {float(gpflops)/2}" + "\n"
  
  with open(os.path.join(fk_bench_dir, "Table-3-float.csv"), "w") as f:
    f.write(floatResultsCSV)

  with open(os.path.join(fk_bench_dir, "Table-3-double.csv"), "w") as f:
    f.write(doubleResultsCSV)

def run_real_world(fk_dir, fk_bench_dir):
  cases = [
           Shape(20, 7, 2, 2),
           Shape(20, 9, 2, 2),
           Shape(50, 9, 2, 2),
           Shape(20, 10, 2, 2),
           Shape(1, 11, 2, 2),

           Shape(10, 2, 64, 64),
           Shape(10, 2, 64, 64),
           Shape(50, 2, 64, 64),
           
           Shape(4, 9, 2, 2),
           Shape(8, 9, 2, 2),
           Shape(16, 9, 2, 2),
           Shape(20, 9, 2, 2),

           Shape(8, 3, 8, 8),
           Shape(8, 3, 8, 8),
           Shape(16, 3, 8, 8),
           Shape(20, 3, 8, 8),

           Shape(1024, 7, 4, 4),
           Shape(1024, 9, 4, 4),
           Shape(256, 7, 8, 8),

           Shape(1, 4, 8, 8),
           Shape(1, 4, 32, 32),

           Shape(1526, 6, 4, 4),
           Shape(156, 3, 8, 8),
           Shape(2967, 7, 4, 4),

           Shape(16, 8, 8, 8),
           Shape(16, 6, 16, 16),
           Shape(16, 5, 32, 32),
           Shape(16, 3, 64, 64),
           ]

  fk_eval = FastKronEval(fk_dir)
  gpEval = GPyTorchEval()
  cogentEval = CogentEval(fk_bench_dir)
  
  resultsCSV = ""
  
  for shape in cases:
    (wofuseflops, _, fuseflops, _) = fk_eval.run_kron(shape, 1, 1, 1, False)
    (gpflops, gptime) = gpEval.run_kron(shape, 1, 1, 1)
    if shape.ps[0] <= 4:
      (cogentflops, cogentime) = (gpflops, gptime)
    else:
      (cogentflops, cogentime) = cogentEval.run_kron(shape)
    resultsCSV += f"{str(shape)} & {fuseflops} & {wofuseflops} & {gpflops} & {cogentflops}" + "\n"
  print("Results\n", resultsCSV)
  with open(os.path.join(fk_bench_dir, "real-world-benchmarks.csv"), "w") as f:
    f.write(resultsCSV)


def run_multi_gpu(fk_dir, fk_bench_dir):
  cases = []
  M_64 = 64
  M_128 = 4
  if total_gpu_memory() <= 16*1024:
    M_64 = M_64/2
    M_128 = M_128/2
  cases += [Shape(M_64, 4, 64, 64)]
  cases += [Shape(M_128, 4, 128, 128)]
  
  # run_command("make gen-multi-gpu-tests-kernel")
  fk_eval = FastKronEval(fk_dir)
  resultsCSV64 = ""
  resultsCSV128 = ""

  smi_output = run_command("nvidia-smi --list-gpus")
  num_gpus = len(smi_output.split("\n"))
  assert f"GPU {num_gpus-1}" in smi_output
  print(f"Found {num_gpus} GPUs")

  for shape in cases:
    GMs = [1, 2, 2, 4, 4]
    GKs = [1, 1, 2, 2, 4]
    
    for j,gpus in enumerate([2**i for i in range(0, int(math.log2(num_gpus))+1)]):
      gm = GMs[j]
      gk = GKs[j]
      shapeGM = Shape(shape.m * gpus, shape.n, shape.ps[0], shape.qs[0])
      (_, _, fkflops, _) = fk_eval.run_kron(shapeGM, gm, gk, 1)
      (_, _, distalflops, _) = fk_eval.run_distal(shapeGM, gm, gk, 1)
      result = f"{str(shapeGM)} & {fkflops} & {distalflops}"
      print(result)
      if shape.ps[0]==64:
        resultsCSV64 += result + "\n"
      else:
        resultsCSV128 += result + "\n"
  
  print(resultsCSV64)
  print(resultsCSV128)
  with open(os.path.join(fk_bench_dir, "multi-gpu-flops-64.csv"), "w") as f:
    f.write(resultsCSV64)
  
  with open(os.path.join(fk_bench_dir, "multi-gpu-flops-128.csv"), "w") as f:
    f.write(resultsCSV128)

def do_evaluation(fk_dir, fk_bench, bench):
  if bench == "Figure-9":
    run_single_gpu_large_M(fk_dir, fk_bench)
  elif bench == "Table-3":
    run_single_gpu_small_M(fk_dir, fk_bench)
  elif bench == "Figure-11":
    run_multi_gpu(fk_dir, fk_bench)
  elif bench == "Figure-10":
    run_real_world(fk_dir, fk_bench)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-fk-dir', required=True, type=str, help='Path to FastKron')
  parser.add_argument('-fk-bench-dir', required=True, type=str, help='Path to FastKron Benchmarks')
  parser.add_argument('-bench', required=True, type=str, help="[Figure-9 | Table-3 | Figure-10 | Figure-11]")

  args = parser.parse_args()

  assert args.bench in ["Figure-9", "Table-3", "Figure-10", "Figure-11"]
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
  
  do_evaluation(os.path.abspath(args.fk_dir), os.path.abspath(args.fk_bench_dir), args.bench)