import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# 1) Define per-dataset attributes 
dataset_attrs = {
    "DROID":              {"T":10,"S":5,"D":0.2,"L":1.0,"M":3,"Q":[0.9,0.8,0.85],"A":0.9,"R":1},
    "Open X-Embodiment":  {"T":15,"S":20,"D":0.5,"L":2.0,"M":4,"Q":[0.8,0.8,0.9,0.7],"A":0.8,"R":1},
    "ALFRED":             {"T":30,"S":10,"D":0.8,"L":3.0,"M":4,"Q":[0.9,0.9,0.9,0.9],"A":0.95,"R":1},
    "RLBench":            {"T":8,"S":6,"D":0.3,"L":1.5,"M":3,"Q":[0.7,0.8,0.7],"A":0.7,"R":0},
    "TEACh":              {"T":12,"S":4,"D":0.6,"L":2.5,"M":3,"Q":[0.8,0.85,0.8],"A":0.85,"R":0},
    "DialFRED":           {"T":25,"S":10,"D":0.75,"L":3.0,"M":4,"Q":[0.85,0.9,0.9,0.85],"A":0.9,"R":1},
    "EmbodiedQA":         {"T":5,"S":2,"D":0.1,"L":1.0,"M":2,"Q":[0.7,0.7],"A":0.6,"R":0},
    "R2R":                {"T":6,"S":3,"D":0.2,"L":1.2,"M":2,"Q":[0.8,0.75],"A":0.7,"R":0},
    "Ego4D":              {"T":20,"S":0,"D":0.4,"L":1.0,"M":3,"Q":[0.9,0.9,0.8],"A":0.9,"R":0},
    "CVDN":               {"T":15,"S":5,"D":0.5,"L":2.0,"M":3,"Q":[0.85,0.8,0.8],"A":0.85,"R":0},
    "CALVIN":             {"T":35,"S":15,"D":0.9,"L":3.5,"M":4,"Q":[0.9,0.85,0.9,0.9],"A":0.9,"R":1},
    "RoboSpatial":        {"T":4,"S":1,"D":0.1,"L":0.5,"M":2,"Q":[0.6,0.65],"A":0.5,"R":0},
    "CoVLA":              {"T":18,"S":8,"D":0.7,"L":2.5,"M":4,"Q":[0.85,0.8,0.85,0.8],"A":0.9,"R":1},
    "AgiBot World":       {"T":40,"S":25,"D":0.95,"L":4.0,"M":3,"Q":[0.8,0.8,0.8],"A":0.8,"R":1},
    "RoboData":           {"T":25,"S":12,"D":0.7,"L":2.5,"M":4,"Q":[0.85,0.9,0.9,0.8],"A":0.9,"R":1},
    "Interleave-VLA":     {"T":18,"S":8,"D":0.6,"L":2.0,"M":4,"Q":[0.8,0.85,0.8,0.75],"A":0.85,"R":1},
    "Iref-VLA":           {"T":22,"S":10,"D":0.65,"L":3.0,"M":5,"Q":[0.9,0.9,0.85,0.9,0.8],"A":0.9,"R":1},
    "RH20T":              {"T":10,"S":4,"D":0.3,"L":1.5,"M":2,"Q":[0.75,0.8],"A":0.8,"R":0},
    "Robo360":            {"T":30,"S":15,"D":0.8,"L":3.5,"M":5,"Q":[0.9,0.85,0.9,0.85,0.9],"A":0.95,"R":1},
    "REASSEMBLE":         {"T":28,"S":12,"D":0.7,"L":3.0,"M":4,"Q":[0.8,0.8,0.85,0.8],"A":0.9,"R":1},
    "RoboCerebra":        {"T":12,"S":6,"D":0.4,"L":2.0,"M":3,"Q":[0.85,0.9,0.85],"A":0.9,"R":0},
    "TLA":                {"T":35,"S":18,"D":0.85,"L":3.8,"M":4,"Q":[0.9,0.9,0.9,0.85],"A":0.9,"R":1},
    "DexGraspNet":        {"T":8,"S":5,"D":0.2,"L":1.2,"M":3,"Q":[0.85,0.85,0.8],"A":0.8,"R":1},
}

# 2) Weights (all =1 for simplicity)
α1, α2, α3, α4 = 1.0, 1.0, 1.0, 1.0
β1, β2, β3, β4 = 1.0, 1.0, 1.0, 1.0

# 3) Compute raw task & modality scores
c_task_raw = {}
c_mod_raw  = {}
for name, a in dataset_attrs.items():
    T, S, D, L = a["T"], a["S"], a["D"], a["L"]
    c_task_raw[name] = α1 * np.log1p(T) + α2 * S + α3 * D + α4 * L
    M = a["M"]; Qm = np.mean(a["Q"]); A, R = a["A"], a["R"]
    c_mod_raw[name]  = β1 * M + β2 * Qm + β3 * A + β4 * R

# 4) Normalize both to [1,5]
def norm15(d):
    arr = np.array(list(d.values()))
    mn, mx = arr.min(), arr.max()
    return {k: 1 + 4*(v-mn)/(mx-mn) for k, v in d.items()}

c_task = norm15(c_task_raw)
c_mod  = norm15(c_mod_raw)

# 5) Point sizes by dataset scale
raw_sizes = [5000,500000,25000,10000,15000,18000,9000,21000,360000,15000,
             23000,5000,12000,20000,20000,15000,10000,5000,25000,18000,
             12000,22000,8000]
names = list(dataset_attrs.keys())
smin, smax = min(raw_sizes), max(raw_sizes)
sizes = {n: (800*(size-smin)/(smax-smin) + 300)*6
         for n, size in zip(names, raw_sizes)}

# 6) Color map
colors = plt.cm.tab20(np.linspace(0,1,len(names)))
col_map = dict(zip(names, colors))

# 7) Plot points
fig, ax = plt.subplots(figsize=(12,8))
for n in names:
    ax.scatter(
        c_task[n], c_mod[n],
        s=sizes[n],
        color=col_map[n],
        alpha=0.7,
        edgecolor=col_map[n],
        linewidth=0.5,
    )

# 8) Axes: x in [0.75,5.25], y in [0.75,5.25] so all normalized scores are visible
ax.set_xlim(0.75, 5.25)
ax.set_ylim(0.75, 5.25)
ax.set_title("Dataset & Benchmark Landscape", fontsize=16, weight='bold')
ax.set_xlabel("Task Complexity", fontsize=12)
ax.set_ylabel("Modality Richness", fontsize=12)

# x‐ticks and labels
ax.set_xticks([1,2,3,4,5])
ax.set_xticklabels(["Very Low","Low","Medium","High","Very High"],
                   rotation=25, ha="right")

# y‐ticks now span exactly the 4 relevant categories: bimodal (2) → pentamodal (5)
ax.set_yticks([2,3,4,5])
ax.set_yticklabels(["Bimodal","Trimodal","Tetramodal","Pentamodal"])
ax.grid(True, linestyle='--', alpha=0.4)

# 9) Legend with small markers
handles = [
    Line2D([], [], marker='o', color='w',
           markerfacecolor=col_map[n], markeredgecolor=col_map[n],
           markersize=6, label=n, linestyle='')
    for n in names
]
ax.legend(
    handles=handles,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.15),
    ncol=4,
    fontsize=8,
    frameon=True
)

plt.tight_layout()
plt.show()
