# t3 [a,320,d,32,b,32,c,32] += sum(e,32) * t2 [a,b,c,e] * v2 [e,d];
import sys
dims = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

def createInputTCCG(rowsA, numFacs, kronRow):
    s = "t3 ["
    s += f"{dims[0]},{rowsA},"
    s += f"{dims[numFacs]},{kronRow},"
    for fac in range(0, numFacs-1):
        s += f"{dims[1+fac]},{kronRow},"
    s = s[:-1]
    s += "] += "
    sumDim = dims[numFacs + 1]
    s += f"sum({sumDim},{kronRow}) * t2 [{dims[0]},"
    for fac in range(0, numFacs-1):
        s += f"{dims[1+fac]},"
    s += f"{sumDim}]"
    s += " * v2 ["
    s += f"{sumDim},{dims[numFacs]}];"
    
    return s

pathToInputTCCG_40 = "../input_strings/tccg/input_tccg_40.in"
rowsA = int(sys.argv[1])
facs = int(sys.argv[2])
kronRows = int(sys.argv[3])
inp = createInputTCCG(rowsA, facs, kronRows)
print("Writing: ", inp)
with open(pathToInputTCCG_40, "w") as f:
    f.write(inp)
print("Calling TC Gen")
import subprocess, sys, os, shutil
currdir = os.getcwd()
os.chdir(currdir+"/..")
(s, o) = subprocess.getstatusoutput("sh ./tccg.sh 40")
if s != 0 or "ERROR" in o:
    print("Error")
    print(o)
    sys.exit(0)

os.chdir(currdir)
shutil.copy(currdir+"/../temp__40.cu",f"./kernels/kernel_{rowsA}_{facs}_{kronRows}.cu")
