import sys
import csv
from common import *

csv_file = sys.argv[1]
pdf_file = sys.argv[2]

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
gp_speedup = []
cogent_speedup = []

for row in filtered_data:
    print(row)
    gp_speedup += [float(row[fk])/float(row[cogent])]
    cogent_speedup += [float(row[fk])/float(row[gp])]

fig, ax2 = plt.subplots(1,1,sharex=True)
# gp_speedup = map(lambda x: x/10 if x >= 5 elif )
p1 = ax2.plot(ind, gp_speedup, color=colors[0], marker='o')
p2 = ax2.plot(ind, cogent_speedup, color=colors[1], marker='s')

# for i, f in enumerate(gp_speedup):
#     ax2.text(i, f, "%.1f"%round(f, 1), color = 'black', fontsize='large', ha='center')

# for i, f in enumerate(cogent_speedup):
#     ax2.text(i, f, "%.1f"%round(f, 1), color = 'black', fontsize='large', ha='center')

# for i, f in enumerate(fastkronwosharedflops):
#     ax2.text(i, f, "%.1f"%round(f, 1), color = 'black', fontsize='large', ha='center')

# for i, f in enumerate(fk_flops):
#     ax2.text(i, f, "%.1f"%round(f, 1), color = 'black', fontsize='large', ha='center')

# for bar1, d in zip(p3, fastkrontimes):
#     ax2.text(bar1.get_x()+bar1.get_width()/2, (bar1.get_height())/2, "%.2f ms"%d, color = 'black', ha = 'center', va = 'center', rotation=90, fontsize='large')

# for bar1, speedup in zip(p3, fastkronspeedup):
#     ax2.text(bar1.get_x()+bar1.get_width()/2+0.04, bar1.get_height()+0.05, r"%.2f$\times$"%(1/speedup), color = 'black', ha = 'center', va = 'center', rotation=0, fontsize='large')

plt.ylabel('Speedup of FastKron')

# plt.xlabel('', fontsize='large')
# plt.title('Contribution by the teams')
# plt.yticks([1,2,4] + )
# plt.yscale("log")
plt.xticks(ind+width, (i for i,row in enumerate(filtered_data)),rotation=0)
plt.legend((p1[0], p2[0]), ('GPyTorch', 'COGENT'),
            loc='upper left', fontsize='large', bbox_to_anchor=(0.0, 1.05),
            ncol=2,columnspacing=1,handlelength=1.7)

FIGURES_DIR = "./"
    
plt.rcParams["font.family"] = "libertine"
#FIGURES_DIR = "./"
fig = plt.gcf()
fig.subplots_adjust(bottom=0.1)
fig.set_size_inches(5.5, 3)

# ax.set_xticks([])
fig.savefig(FIGURES_DIR+pdf_file,bbox_inches='tight',pad_inches=0)
plt.show()