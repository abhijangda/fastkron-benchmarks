import os
import subprocess
import re

class Shape:
  def __init__(self, m, n, p, q):
    self.m = m
    self.n = n
    self.ps = [p for i in range(0, n)]
    self.qs = [q for i in range(0, n)]

  def __repr__(self):
    return f"{self.m}_{self.ps[0]}x{self.qs[0]}^{self.n}"

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

  def gen_kernels(shape, distKernels):
    run_command("python3 src/gen_tuner_kernels.py -distinct-factors " + \
                str(shape.n) + " " + " ".join([f"{pq[0]},{pq[1]}" for pq in zip(shape.ps, shape.qs)]) + \
                (" -dist-kernels " if distKernels else ""))

  def build_kron():
    run_command("make kron -j")

  def run_kron(shape, GM, GK, LocalKrons):
    with Executor(self.fk_dir) as executor:
      gen_kernels(shape, False)
      build_kron()
      kron = f"./kron -m {shape.m} -n {shape.n} -p {shape.ps[0]} -q {shape.qs[0]} -r 20 -w 10 -t float --tune"
      if GM * GK != 1:
        kron += f" --gpus {GM*GK} --GM {GM} --GK {GK} --gpuLocalKrons {LocalKrons}"

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

      return (shape, GM*GK, wofuse, wofusetime, fused, fusedtime)

def run_single_gpu(fk_dir, fk_bench_dir):
  M = 1024
  cases = [Shape(M, 5, 8, 8),     Shape(M, 6, 8, 8),
           Shape(M, 4, 16, 16),   Shape(M, 5, 16, 16),
           Shape(M, 3, 32, 32),   Shape(M, 4, 32, 32),
           Shape(M, 2, 64, 64),   Shape(M, 3, 64, 64),
           Shape(M, 2, 128, 128), Shape(M, 3, 128, 128)]
  fk_eval = FastKronEval(fk_dir)
  for shape in cases:
    fk_eval.run_kron(shape, 1, 1, 1)

def run_single_gpu_small():
  M = 16
  cases = [Shape(M, 8, 8, 8),
           Shape(M, 6, 16, 16),
           Shape(M, 5, 32, 32),
           Shape(M, 4, 64, 64),
          #  Shape(M, 3, 128, 128)
           ]

  for shape in cases:
    gen_kernels(shape, False)
    build_kron()
    run_kron(shape, 1, 1, 1)

def multi_gpu():
  cases = []
  M_64 = 64
  cases += [Shape(M_64, 4, 64, 64)]
  M_128 = 4
  cases += [Shape(M_128, 4, 128, 128)]
  
  # run_command("make gen-multi-gpu-tests-kernel")

  for shape in cases:
    GMs = [1, 2, 2, 4, 4]
    GKs = [1, 1, 2, 2, 4]
    gen_kernels(shape, True)
    build_kron()
    for j,gpus in enumerate([1, 2, 4, 8, 16]):
      gm = GMs[j]
      gk = GKs[j]
      shapeGM = Shape(shape.m * gm, shape.n, shape.ps[0], shape.qs[0])
      LocalKrons = shapeGM.n if gk == 1 else shapeGM.n - 2
      run_kron(shapeGM, gm, gk, LocalKrons)


def do_evaluation(fk_dir, fk_bench):
  run_single_gpu(fk_dir, fk_bench)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-fk', required=True, type=str, help='Path to FastKron')
  parser.add_argument('-fk-bench', required=True, type=str, help='Path to FastKron Benchmarks')

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
  
  do_evaluation(parser.fk_dir, parser.fk_bench)