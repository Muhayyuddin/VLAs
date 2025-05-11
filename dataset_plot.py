import matplotlib.pyplot as plt
import numpy as np

# Dataset positions
datasets = {
    "DROID": (1.5, 4.5),
    "Open X-Embodiment": (2, 4),
    "ALFRED": (4.5, 4.5),
    "RLBench": (2, 3),
    "TEACh": (2.5, 3),
    "DialFRED": (2.7, 3),
    "EmbodiedQA": (1.2, 2.5),
    "R2R": (1.5, 2),
    "Ego4D": (2.2, 3.2),
    "CVDN": (2.5, 2.8),
    "CALVIN": (4, 3.5),
    "RoboSpatial": (1, 2),
    "CoVLA": (3, 3),
    "AgiBot World": (4.5, 3)
}

# Raw sizes
raw_sizes = {
    "DROID": 500,
    "Open X-Embodiment": 450,
    "ALFRED": 1500,
    "RLBench": 400,
    "TEACh": 350,
    "DialFRED": 300,
    "EmbodiedQA": 300,
    "R2R": 300,
    "Ego4D": 600,
    "CVDN": 350,
    "CALVIN": 450,
    "RoboSpatial": 300,
    "CoVLA": 350,
    "AgiBot World": 450
}

# Normalize sizes
min_size = min(raw_sizes.values())
max_size = max(raw_sizes.values())
scaled_sizes = {
    name: (800 * (size - min_size) / (max_size - min_size) + 300)*4
    for name, size in raw_sizes.items()
}

# Assign distinct colors
colors = plt.cm.tab20(np.linspace(0, 1, len(datasets)))
dataset_colors = dict(zip(datasets.keys(), colors))

fig, ax = plt.subplots(figsize=(12, 8))

# Plot datasets
for name, (x, y) in datasets.items():
    ax.scatter(x, y, s=scaled_sizes[name],
               color=dataset_colors[name],
               alpha=0.7,
               edgecolors=dataset_colors[name],
               linewidth=1.0,
               label=name)

# Titles and axis labels
ax.set_title("Dataset & Benchmark Landscape", fontsize=16, weight='bold')
ax.set_xlabel("Task Complexity", fontsize=12)
ax.set_ylabel("Modality Richness", fontsize=12)
ax.set_xlim(0.5, 5)
ax.set_ylim(1, 5)
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_yticks([1, 2, 3, 4, 5])
ax.set_xticklabels(["Single-step Actions", "", "", "", "Multi-step Planning"])
ax.set_yticklabels(["Single Modality", "", "", "", "Video + Lang + State"])
ax.grid(True, linestyle='--', alpha=0.5)

# Custom legend
handles = [plt.Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=dataset_colors[name], label=name,
                      markersize=10, markeredgecolor=dataset_colors[name]) for name in datasets]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=3)

plt.tight_layout()
plt.show()
