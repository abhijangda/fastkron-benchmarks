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
  def __init__(self, fk_dir, backend, mode, elemtype):
    self.fk_dir = fk_dir
    self.last_shape = None
    self.backend = backend
    self.tuningmode = mode
    self.built = False
    self.elemtype = elemtype

  def gen_kernels(self, shape, opX, opF, distKernels):
    if self.tuningmode == 'FullTune':
      run_command("python3 src/gen_tuner_kernels.py -backend cuda -archs ampere volta -distinct-factors " + \
                  str(shape.n) + " " + " ".join([f"{pq[0]},{pq[1]}" for pq in zip(shape.ps, shape.qs)]) + \
                  " -opX " + opX + " -opF " + opF + \
                  (" -dist-kernels " if distKernels else "") + \
                  " -backend " + self.backend + " -types " + self.elemtype + " -opt-levels 3")
    elif self.tuningmode == 'FastTune' or self.tuningmode == 'NoTune':
      pass #run_command("cd build/ && make gen-single-gpu-kernels")

  def setup_cmake(self):
    if self.built == True:
      return
    d = os.getcwd()
    if os.path.exists('build/'):
      shutil.rmtree('build/')
    os.mkdir('build/')
    os.chdir('build/')
    if self.backend == "cuda":
      backend_flags = '-DCMAKE_CUDA_FLAGS="-Xptxas -v -O3" -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;80"'
    elif self.backend == "x86":
      backend_flags = "-DENABLE_X86=ON"
    if self.tuningmode == "FullTune":
      backend_flags += " -DFULL_TUNE=ON"
    run_command('cmake .. ' + backend_flags)
    os.chdir(d)

  def build_kron(self):
    run_command(f"cd build && make benchmark_{self.backend} -j")

  def run_kron(self, shape, GM, GK, LocalKrons, opX, opF, callnofuse=True):
    kron = f"cd build && {'TUNE=0' if self.tuningmode=='NoTune' else ''} ./tests/benchmarks/benchmark_{self.backend} -m {shape.m} -n {shape.n} -p {shape.ps[0]} -q {shape.qs[0]} -r 10 -w {50 if self.tuningmode=='NoTune' else 20} -t {self.elemtype} --tune --opx {opX} --opf {opF}"
    if GM * GK != 1:
      kron += f" --gpus {GM*GK} --GM {GM} --GK {GK} --gpuLocalKrons {LocalKrons}"
    kron += " --backend " + self.backend

    o = run_command(kron + " --fuse")
    fused = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
    fusedtime = re.findall(r"Time: ([\d\.]+) ms", o)[0]
    if shape.ps[0] <= 32 and shape.ps[0] == shape.qs[0]:
      o = run_command(kron)
      wofuse = re.findall(r"GFLOPS\: ([\d\.]+)", o)[0]
      wofusetime = re.findall(r"Time: ([\d\.]+) ms", o)[0]
    else:
      wofuse = fused
      wofusetime = fusedtime

    if GM*GK == 1:
      return (shape, float(wofuse), float(wofusetime), float(fused), float(fusedtime))
    else:
      return (shape, GM, GK, wofuse, fused)

  def run_single_gpu(self, shape, opX, opF):
    with Executor(self.fk_dir) as executor: 
      if self.built == False:
        self.setup_cmake()
        self.gen_kernels(shape, opX, opF, False)
        self.build_kron()
        if self.tuningmode == 'FastTune' or self.tuningmode == 'NoTune':
          self.built = True
      return self.run_kron(shape, 1, 1, 1, opX, opF)
  
  def setup_multi_gpu(self, shape):
    with Executor(self.fk_dir) as executor:
      self.gen_kernels(shape, "N", "N", True)
      self.setup_cmake()
      self.build_kron()
  
  def run_multi_gpu(self, shape, GM, GK, LocalKrons):
    with Executor(self.fk_dir) as executor:
      return self.run_kron(shape, GM, GK, LocalKrons, "N", "N")

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
  def __init__(self, backend, elemtype):
    self.backend = backend
    if elemtype == "float":
      self.elemtype = torch.float
    elif elemtype == "double":
      self.elemtype = torch.double
    elif elemtype == "half":
      self.elemtype = torch.half
    if self.backend == "x86" and "OMP_NUM_THREADS" in os.environ:
      torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))

  def run_single_gpu(self, shape):
    r = self._run_kron(shape)
    torch.cuda.empty_cache()
    return r

  def _run_kron(self, shape):
    import gpytorch as gp
    import torch
    factors = []
    for p,q in zip(shape.ps, shape.qs):
      f = torch.ones(p, q, dtype=self.elemtype)
      if self.backend == 'cuda':
        f = f.cuda()
      factors += [f] 
    x = torch.ones(shape.m, shape.k, dtype=self.elemtype)
    if self.backend == 'cuda':
      x = x.cuda()
    kp = gp.lazy.KroneckerProductLazyTensor(*factors)
    def run_case(r):
        t1 = time.time()
        for i in range(r):
            y = x @ kp
        torch.cuda.synchronize()
        t2 = time.time()
        return (t2-t1)*1000/r
    
    run_case(10)
    t = min(run_case(5), run_case(5), run_case(5), run_case(5), run_case(5))
    flops = shape.flops()/(t/1e3)
    return (flops/1e9,t)

class CogentEval:
  def __init__(self, fk_bench_dir):
    self.fk_bench_dir = fk_bench_dir
  
  def run_kron(self, shape):
    with Executor(os.path.join(self.fk_bench_dir, "TC-CGO2019/cogent/kron")) as exector:
      ps_0 = shape.ps[0]
      o = run_command(f"python3 gen_kernels.py {shape.m} {shape.n} {ps_0}")
      o = run_command(f'nvcc -O3 -std=c++11 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_70,code=sm_70 mains/main_{shape.n}_facs.c kernels/kernel_{shape.m}_{shape.n}_{ps_0}.cu -Xptxas "-v " -o run_temp')
      o = run_command(f"./run_temp {shape.m} {ps_0}")
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

class CuTensorEval:
  def __init__(self, fk_bench_dir):
    self.fk_bench_dir = fk_bench_dir
  
  def run_kron(self, shape):
    with Executor(os.path.join(self.fk_bench_dir, "cutensor")):
      o = run_command(f"make")
      o = run_command(f"./contraction {shape.m} {shape.n} {shape.ps[0]} {shape.qs[0]}")
      print(o)
      return self.parse_output(shape, o)
    
  def parse_output(self, shape, output):
    try:
      ft = re.findall(r'([\d\.]+)', output)
      f,t = ft[0], ft[1]
      f,t = float(f),float(t)
      return (f, t)
    except:
      print(output)
      return None

class TCCGEval:
  def __init__(self, fk_bench_dir):
    self.fk_bench_dir = fk_bench_dir
    self.kron_matmul = "kron-matmul.tccg"

  def write_kron_matmul(self, shape):
    with open(f"{self.fk_bench_dir}/tccg/{self.kron_matmul}", "w") as f:
      s = "C[m,c,a] = A[m,a,b] * B[b,c]\n"
      s += f"m = {shape.m}\n"
      s += f"a = {shape.ps[0]**(shape.n-1)}\n"
      s += f"b = {shape.ps[0]}\n"
      s += f"c = {shape.qs[0]}\n"
      f.write(s)

  def run_kron(self, shape):
    with Executor(os.path.join(self.fk_bench_dir, "tccg")) as exe:
      self.write_kron_matmul(shape)
      run_command(f"rm -rf {exe.exec_dir}/tccg_implementations")
      run_command(f"rm -f {exe.exec_dir}/tccg/tccg.db")
      ld_library_path = f"LD_LIBRARY_PATH={exe.exec_dir}/hptt/lib:{exe.exec_dir}/tcl/lib:$LD_LIBRARY_PATH"
      o = run_command(f"{ld_library_path} TCCG_ROOT=`pwd` python2 tccg/tccg.py --arch=avx2 --numThreads=$OMP_NUM_THREADS --floatType=s kron-matmul.tccg --verbose")
      return self.parse_output(shape, o)

  def parse_output(self, shape, output):
    try:
      f = [float(f) for f in re.findall(r"attained: ([\d\.]*) GFLOPS", output)]
      f = max(f)
      return (f, 0)
    except Exception as e:
      print(e)
      return None

def run_single_node(cases, csv, fk_dir, fk_bench_dir, device):
  resultsCSV = ""
  fk_eval = FastKronEval(fk_dir, device, "FastTune", "float")
  gpEval = GPyTorchEval(device, "float")

  if device == "cuda":
    cogentEval = CogentEval(fk_bench_dir)
    cutensorEval = CuTensorEval(fk_bench_dir)
  elif device == "x86":
    tccgEval = TCCGEval(fk_bench_dir)

  for shape in cases:
    result = f"{str(shape)}"
    (_, wofuseflops, _, fuseflops, _) = fk_eval.run_single_gpu(shape, "N", "N")
    (gpflops, gptime) = gpEval.run_single_gpu(shape)
    result = f"{str(shape)} & {fuseflops} & {wofuseflops} & {gpflops} & "
    if device == "cuda":
      if shape.ps[0] <= 4:
        (cogentflops, cogentime) = (gpflops, gptime)
      else:
        (cogentflops, cogentime) = cogentEval.run_kron(shape)
      (cutensorflops, cutensortime) = cutensorEval.run_kron(shape)
      result += f"{cogentflops} & {cutensorflops}"
    elif device == "x86":
      result += f"{tccgEval.run_kron(shape)[0]}"

    print("TFLOPs", result)
    resultsCSV += result + "\n"

  with open(csv, "w") as f:
    f.write(resultsCSV)

def run_large_M(fk_dir, fk_bench_dir, device):
  device = device.lower()
  assert device.lower() in ["cuda", "x86"]
  M = 1024
  M2 = 320
  if device == "cuda" and total_gpu_memory() <= 16*1024:
    M = M/2
    M2 = M2/2
  elif device == "x86":
    M = 256
    M2 = 128

  cases = [Shape(M, 5, 8, 8),     Shape(M, 6, 8, 8),
           Shape(M, 4, 16, 16),   
           Shape(M, 5, 16, 16),
           Shape(M, 3, 32, 32),   Shape(M, 4, 32, 32),
           Shape(M, 2, 64, 64),   Shape(M, 3, 64, 64),
           Shape(M, 2, 128, 128), Shape(M2, 3, 128, 128)
           ]
  run_single_node(cases, os.path.join(fk_bench_dir, f"single-{device}-flops.csv"), 
                  fk_dir, fk_bench_dir, device)

def run_single_gpu_small_M(fk_dir, fk_bench_dir, device):
  M = 16
  if total_gpu_memory() <= 16*1024:
    M = M/2
  cases = [Shape(M, 8, 8, 8),
           Shape(M, 6, 16, 16),
           Shape(M, 5, 32, 32),
           Shape(M, 4, 64, 64)]
  run_single_node(cases, os.path.join(fk_bench_dir, "Table-3-float.csv"), 
                fk_dir, fk_bench_dir, device)
  # with open(os.path.join(fk_bench_dir, "Table-3-double.csv"), "w") as f:
  #   f.write(doubleResultsCSV)

def run_real_world(fk_dir, fk_bench_dir, device):
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

  run_single_node(cases, os.path.join(fk_bench_dir, "real-world-benchmarks.csv"), 
                  fk_dir, fk_bench_dir, device)

def run_multi_gpu(fk_dir, fk_bench_dir):
  cases = []
  M_64 = 64
  M_128 = 4
  if total_gpu_memory() <= 16*1024:
    M_64 = M_64/2
    M_128 = M_128/2
  cases += [Shape(M_64, 4, 64, 64)]
  cases += [Shape(M_128, 4, 128, 128)]
  scaling = "strong"
  # run_command("make gen-multi-gpu-tests-kernel")
  resultsCSV64 = ""
  resultsCSV128 = ""

  smi_output = run_command("nvidia-smi --list-gpus")
  num_gpus = len(smi_output.split("\n"))
  assert f"GPU {num_gpus-1}" in smi_output
  print(f"Found {num_gpus} GPUs")

  for shape in cases:
    GMs = [1, 2, 2, 4, 4]
    GKs = [1, 1, 2, 2, 4]
    fk = FastKronEval(fk_dir, "cuda", "FullTune", "float")
    fk.setup_multi_gpu(shape)
    for j,gpus in enumerate([1, 2, 4, 8]):
      gm = GMs[j]
      gk = GKs[j]
      shapeGM = Shape(shape.m * (gpus if scaling == "weak" else 1), shape.n, shape.ps[0], shape.qs[0])
      LocalKrons = shapeGM.n if gk == 1 else shapeGM.n - 2
      (_, _, fkflops, _) = fk.run_multi_gpu(shapeGM, gm, gk, LocalKrons, "N", "N")      
      result = f"{str(shapeGM)} & {fkflops}"
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
    run_large_M(fk_dir, fk_bench, "cuda")
  elif bench == "Table-3":
    run_single_gpu_small_M(fk_dir, fk_bench, "cuda")
  elif bench == "Figure-11":
    run_multi_gpu(fk_dir, fk_bench)
  elif bench == "Figure-10":
    run_real_world(fk_dir, fk_bench, "cuda")
  elif bench == "x86":
    run_large_M(fk_dir, fk_bench, "x86")

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-fk-dir', required=True, type=str, help='Path to FastKron')
  parser.add_argument('-fk-bench-dir', required=True, type=str, help='Path to FastKron Benchmarks')
  parser.add_argument('-bench', required=True, type=str, help="[Figure-9 | Table-3 | Figure-10 | Figure-11]")

  args = parser.parse_args()

  assert args.bench in ["Figure-9", "Table-3", "Figure-10", "Figure-11", "x86"]
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
