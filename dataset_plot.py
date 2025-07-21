import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# 1) Define per-dataset attributes (updated with corrected values and new datasets)
# Descriptions:
# T: Number of distinct tasks / skill types (higher is broader)
# S: Scene diversity (number of unique environments)
# D: Task difficulty (normalized, 0–1, higher is more challenging)
# L: Task/episode length or complexity (normalized, higher = longer)
# M: Number of modalities (vision, lang, proprioception, depth, audio, etc.)
# Q: List of quality/success scores (per modality or overall)
# A: Average annotation or benchmark score (0–1)
# R: Real-robot validation (1 = yes, 0 = sim-only)
dataset_attrs = {
    "DROID":              {"T":10, "S":5,  "D":0.2,  "L":1.0, "M":3, "Q":[0.9,0.8,0.85],           "A":0.9,  "R":1},
    "Open X-Embodiment":  {"T":15, "S":20, "D":0.5,  "L":2.0, "M":4, "Q":[0.8,0.8,0.9,0.7],     "A":0.8,  "R":1},
    "ALFRED":             {"T":30, "S":10, "D":0.8,  "L":3.0, "M":4, "Q":[0.9,0.9,0.9,0.9],     "A":0.95, "R":1},
    "RLBench":            {"T":8,  "S":6,  "D":0.3,  "L":1.5, "M":3, "Q":[0.7,0.8,0.7],         "A":0.7,  "R":0},
    "TEACh":              {"T":12, "S":4,  "D":0.6,  "L":2.5, "M":3, "Q":[0.8,0.85,0.8],       "A":0.85, "R":0},
    "DialFRED":           {"T":25, "S":10, "D":0.75, "L":3.0, "M":4, "Q":[0.85,0.9,0.9,0.85],  "A":0.9,  "R":1},
    "EmbodiedQA":         {"T":5,  "S":2,  "D":0.1,  "L":1.0, "M":2, "Q":[0.7,0.7],             "A":0.6,  "R":0},
    "R2R":                {"T":6,  "S":3,  "D":0.2,  "L":1.2, "M":2, "Q":[0.8,0.75],            "A":0.7,  "R":0},
    "Ego4D":              {"T":20, "S":0,  "D":0.4,  "L":1.0, "M":3, "Q":[0.9,0.9,0.8],         "A":0.9,  "R":0},
    "CVDN":               {"T":15, "S":5,  "D":0.5,  "L":2.0, "M":3, "Q":[0.85,0.8,0.8],        "A":0.85, "R":0},
    "CALVIN":             {"T":35, "S":15, "D":0.9,  "L":3.5, "M":4, "Q":[0.9,0.85,0.9,0.9],   "A":0.9,  "R":1},
    "RoboSpatial":        {"T":4,  "S":1,  "D":0.1,  "L":0.5, "M":2, "Q":[0.6,0.65],           "A":0.5,  "R":0},
    "CoVLA":              {"T":18, "S":8,  "D":0.7,  "L":2.5, "M":4, "Q":[0.85,0.8,0.85,0.8], "A":0.9,  "R":1},
    "AgiBot World":       {"T":30, "S":25, "D":0.95, "L":4.0, "M":3, "Q":[0.8,0.8,0.8],       "A":0.8,  "R":1},
    "RoboData":           {"T":25, "S":12, "D":0.7,  "L":2.5, "M":4, "Q":[0.85,0.9,0.9,0.8], "A":0.9,  "R":1},
    "Interleave-VLA":     {"T":18, "S":8,  "D":0.6,  "L":2.0, "M":4, "Q":[0.8,0.85,0.8,0.75], "A":0.85, "R":1},
    "Iref-VLA":           {"T":22, "S":10, "D":0.65, "L":3.0, "M":5, "Q":[0.9,0.9,0.85,0.9,0.8], "A":0.9,  "R":1},
    "RH20T":              {"T":10, "S":4,  "D":0.3,  "L":1.5, "M":2, "Q":[0.75,0.8],           "A":0.8,  "R":0},
    "Robo360":            {"T":30, "S":15, "D":0.8,  "L":3.5, "M":5, "Q":[0.9,0.85,0.9,0.85,0.9],"A":0.95, "R":1},
    "REASSEMBLE":         {"T":28, "S":12, "D":0.7,  "L":3.0, "M":4, "Q":[0.8,0.8,0.85,0.8],   "A":0.9,  "R":1},
    "RoboCerebra":        {"T":12, "S":6,  "D":0.4,  "L":2.0, "M":3, "Q":[0.85,0.9,0.85],      "A":0.9,  "R":0},
    "TLA":                {"T":35, "S":18, "D":0.85, "L":3.8, "M":4, "Q":[0.9,0.9,0.9,0.85],  "A":0.9,  "R":1},
    "Kaiwu":              {"T":30, "S":20, "D":0.7,  "L":4.0, "M":7, "Q":[0.9]*7,               "A":0.9,  "R":1},  # Source: arXiv:2503.05231
    "RefSpatial-Bench":   {"T":2, "S":3, "D":1.0,  "L":4.0, "M":2, "Q":[0.4696, 0.0582, 0.2287, 0.2191, 0.4577, 0.47, 0.52, 0.52, 0.2421, 0.0431, 0.0927, 0.1285, 0.1474, 0.48, 0.53, 0.54], "A":0.9,  "R":1},  # Source: arXiv:2506.04308
}

# 2) Weights
α1, α2, α3, α4 = 1.0, 1.0, 1.0, 1.0
β1, β2, β3, β4 = 1.0, 1.0, 1.0, 1.0

# 3) Compute raw task & modality scores
c_task_raw = {}
c_mod_raw  = {}
for name, a in dataset_attrs.items():
    T, S, D, L = a["T"], a["S"], a["D"], a["L"]
    c_task_raw[name] = α1 * np.log1p(T) + α2 * S + α3 * D + α4 * L
    M = a["M"]
    Qm = np.mean(a["Q"])
    A, R = a["A"], a["R"]
    c_mod_raw[name]  = β1 * M + β2 * Qm + β3 * A + β4 * R

# 4a) Normalize task to [1,5]
def norm15(d):
    arr = np.array(list(d.values()))
    mn, mx = arr.min(), arr.max()
    return {k: 1 + 4*(v-mn)/(mx-mn) for k, v in d.items()}

# 4b) Normalize modality to [2,5]
def norm25(d):
    arr = np.array(list(d.values()))
    mn, mx = arr.min(), arr.max()
    return {k: 2 + 3*(v-mn)/(mx-mn) for k, v in d.items()}

c_task = norm15(c_task_raw)
c_mod  = norm25(c_mod_raw)

# 5) Point sizes by dataset scale 
raw_sizes = [
    5000, 500000, 25025, 10000, 15000, 18000, 9000, 21000, 360000,
    15000, 23000, 5000, 12000, 20000, 20000, 15000, 10000, 5000,
    25000, 18000, 12000, 22000, 8000, 11664
]
names = list(dataset_attrs.keys())
smin, smax = min(raw_sizes), max(raw_sizes)
sizes = {n: (800*(sz-smin)/(smax-smin) + 300)*6 for n, sz in zip(names, raw_sizes)}

# 6) Color map
colors = plt.cm.tab20(np.linspace(0, 1, len(names)))
col_map = dict(zip(names, colors))
# override to distinguish R2R and Ego4D
col_map["R2R"] = "tab:red"
col_map["Ego4D"] = "tab:purple"

# 7) Plot (no text labels on bubbles)
fig, ax = plt.subplots(figsize=(12, 8))
for n in names:
    x, y = c_task[n], c_mod[n]
    ax.scatter(
        x, y,
        s=sizes[n],
        facecolor=col_map[n],
        edgecolor=col_map[n],
        alpha=0.7,
        linewidth=0.5,
    )

# 8) Axes settings
ax.set_xlim(0.75, 5.25)
ax.set_ylim(1.75, 5.25)
ax.set_title("Dataset & Benchmark Landscape", fontsize=16, weight='bold')
ax.set_xlabel("Task Complexity", fontsize=12)
ax.set_ylabel("Modality Richness", fontsize=12)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["Very Low", "Low", "Medium", "High", "Very High"], rotation=25, ha="right")
ax.set_yticks([2, 3, 4, 5])
ax.set_yticklabels(["Minimal", "Moderate", "Rich", "Comprehensive"])
ax.grid(True, linestyle='--', alpha=0.4)

# 9) Legend
handles = [
    Line2D([], [], marker='o', color='w',
           markerfacecolor=col_map[n], markeredgecolor=col_map[n],
           markersize=6, label=n, linestyle='')
    for n in names
]
ax.legend(handles=handles,
          loc='upper center',
          bbox_to_anchor=(0.5, -0.15),
          ncol=4,
          fontsize=8,
          frameon=True)

plt.tight_layout()
plt.show()
