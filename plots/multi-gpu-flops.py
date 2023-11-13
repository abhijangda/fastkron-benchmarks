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

def parse_m(s):
    return s[:s.find('_')]
    
import matplotlib.pyplot as plt
import numpy as np

ind = np.arange(len(filtered_data))
width = 0.1
fk = 1
distal = 2

# fig = plt.subplots(figsize =(10, 7))
fk_flops = []
distal_flops = []

for row in filtered_data:
    fk_flops += [float(row[fk])/1e3]
    distal_flops += [float(row[distal])/1e3]
    
fig, ax2 = plt.subplots(1,1,sharex=True)

p1 = ax2.plot(ind, distal_flops, color=colors[0], marker='o')
p2 = ax2.plot(ind, fk_flops, color=colors[1], marker='x')

for i, f in enumerate(fk_flops):
    ax2.text(i, f, "%.1f"%round(f, 1), color = 'black', fontsize='large', ha='center')

plt.ylabel('TFLOPS')

plt.xlabel('Kron-Matmul of M=1024 and diverse P$^N$ values', fontsize='large')
# plt.title('Contribution by the teams')
plt.xticks(ind+width, [f"{parse_m(row[0])}, {2**g}" for g,row in enumerate(filtered_data)])
plt.legend((p1[0], p2[0]), ('DISTAL', 'FastKron'),
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