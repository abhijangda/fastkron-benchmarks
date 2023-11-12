import sys
import csv
from common import *

csv_file = sys.argv[1]

filtered_data = []

with open(csv_file, 'r') as f:
    csv_reader = csv.reader(f,delimiter='&')
    for i, row in enumerate(csv_reader):
        row_new = []
        for e in row:
            if e != "":
                row_new.append(e.strip())
        row = row_new
        if row == [] or row == None:
            continue
        filtered_data += [row]

print(filtered_data)

def parse_p_n(s):
    p = s[s.find('x')+1:s.find('^')]
    n = s[s.find('^')+1:]
    return f"${p}^{n}$"

import matplotlib.pyplot as plt
import numpy as np

ind = np.arange(len(filtered_data))
width = 0.1
fk = 1
fk_wo_fused = 2
cogent = 3
gp = 4

# fig = plt.subplots(figsize =(10, 7))
fk_flops = []
fk_wo_fused_flops = []
gp_flops = []
cogent_flops = []

for row in filtered_data:
    print(row)
    fk_flops += [float(row[fk])/1e3]
    fk_wo_fused_flops += [float(row[fk_wo_fused])/1e3]
    cogent_flops += [float(row[cogent])/1e3]
    gp_flops += [float(row[gp])/1e3]
    
fig, ax2 = plt.subplots(1,1,sharex=True)

p1 = ax2.plot(ind, gp_flops, color=colors[0], marker='o')
p2 = ax2.plot(ind, cogent_flops, color=colors[1], marker='s')
p4 = ax2.plot(ind, fk_wo_fused_flops, color=colors[3], marker='d')
p3 = ax2.plot(ind, fk_flops, color=colors[2], marker='x')

# for i, f in enumerate(pytorchflops):
#     ax2.text(i, f, "%.1f"%round(f, 1), color = 'black', fontsize='large', ha='center')

# for i, f in enumerate(cogentflops):
#     ax2.text(i, f, "%.1f"%round(f, 1), color = 'black', fontsize='large', ha='center')

# for i, f in enumerate(fastkronwosharedflops):
#     ax2.text(i, f, "%.1f"%round(f, 1), color = 'black', fontsize='large', ha='center')

for i, f in enumerate(fk_flops):
    ax2.text(i, f, "%.1f"%round(f, 1), color = 'black', fontsize='large', ha='center')

# for bar1, d in zip(p3, fastkrontimes):
#     ax2.text(bar1.get_x()+bar1.get_width()/2, (bar1.get_height())/2, "%.2f ms"%d, color = 'black', ha = 'center', va = 'center', rotation=90, fontsize='large')

# for bar1, speedup in zip(p3, fastkronspeedup):
#     ax2.text(bar1.get_x()+bar1.get_width()/2+0.04, bar1.get_height()+0.05, r"%.2f$\times$"%(1/speedup), color = 'black', ha = 'center', va = 'center', rotation=0, fontsize='large')

plt.ylabel('TFLOPS')

plt.xlabel('Kron-Matmul of M=1024 and diverse P$^N$ values', fontsize='large')
# plt.title('Contribution by the teams')
plt.yticks([0,2,4,6,8,10,12,14])
plt.xticks(ind+width, (parse_p_n(row[0]) for row in filtered_data),rotation=45)
plt.legend((p1[0], p2[0], p3[0], p4[0]), ('GPyTorch', 'COGENT', 'FastKron', 'FastKron-wo-Fuse'),
            loc='upper left', fontsize='large', bbox_to_anchor=(0.0, 1.05),
            ncol=2,columnspacing=1,handlelength=1.7)

FIGURES_DIR = "./"
    
plt.rcParams["font.family"] = "libertine"
#FIGURES_DIR = "./"
fig = plt.gcf()
fig.subplots_adjust(bottom=0.1)
fig.set_size_inches(5.5, 3)

# ax.set_xticks([])
figure_file = "single-gpu-float-flops.pdf"
fig.savefig(FIGURES_DIR+figure_file,bbox_inches='tight',pad_inches=0)
plt.show()