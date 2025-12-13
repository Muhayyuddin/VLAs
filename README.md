# Vision–Language–Action (VLA) Models in Robotics

This repository was developed alongside the paper [Vision Language Action Models in Robotic Manipulation: A Systematic Review](https://arxiv.org/pdf/2507.10672v1) and provides a living catalog of:

- **Dataset Benchmarking Code**  
  Code to benchmark the datasets based on the task complexity and modality richness.
  
- **VLA Models**  
  Key vision–language–action models that are used in the review, with links to the original papers.
- **Datasets**  
  Major benchmarks and large‑scale collections used to train and evaluate VLA systems, including QA/navigation datasets, manipulation demonstrations, and multimodal embodiment data.
- **Simulators**  
  Widely adopted simulation platforms for generating VLA data—spanning photorealistic navigation, dexterous manipulation, multi‑robot coordination, and more—each linked to its official website.

We aim to keep this list up to date as new VLA models, datasets, and simulation tools emerge. Contributions and pull requests adding recently published work or tooling are most welcome!  

---

## Table of Contents
- [Dataset Benchmarking Code](#Dataset-Benchmarking-Code)
- [VLA Models Evaluation & Visualization](#vla-models-evaluation--visualization)
- [VLA Models](#vla-models)  
- [Datasets](#datasets)  
- [Simulators](#simulators)
- [Reference for Citation](#reference)



---
# Dataset Benchmarking Code 
Benchmarking VLA Datasets by Task Complexity and Modality Richness. Each bubble represents a VLA dataset, positioned according to its normalized task-complexity score (x-axis) and its modality-richness score (y-axis). The bubble area is proportional to the dataset scale that is number of annotated episodes or interactions. 

![Dataset Benchmarking](https://github.com/Muhayyuddin/VLAs/blob/main/benchmarkdataset.png)

[Code](https://github.com/Muhayyuddin/VLAs/blob/main/dataset_plot.py)

---

# VLA Models Evaluation & Visualization

This repository includes a comprehensive analysis and visualization suite for evaluating Vision-Language-Action (VLA) models. The analysis covers multiple aspects of VLA model performance, architecture components, and theoretical foundations through detailed visualizations and statistical analysis.

## Key Analysis Features

- **Forest Plot Analysis**: Comparative performance metrics across different VLA models
- **Encoder Component Analysis**: Deep dive into visual encoder architectures and their impact on performance  
- **Domain Analysis**: Cross-domain performance evaluation and transfer capabilities
- **Fusion Theory Visualization**: Analysis of multimodal fusion strategies in VLA models
- **VLA-FEB Score Distribution**: Histogram analysis of the VLA Fusion-Encoder-Backbone composite metric

## Quick Start for Analysis

### Setup Environment
```bash
cd Plot_script/
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Generate All Visualizations
```bash
python final_plots.py
```

## Key Metrics & Files

- **`new_vla_models.csv`**: Complete dataset with 101 VLA models and comprehensive evaluation metrics
- **`top75.csv`**: Curated subset featuring VLA-FEB component scores (CMAS, E_fusion, R2S, GI)
- **VLA-FEB Score**: Composite metric combining Cross-Modal Alignment Score (CMAS), Fusion Energy (E_fusion), Real-to-Sim Transfer (R2S), and Generalization Index (GI)
- **Adjusted Success Rate**: Normalized task success rates on a 0-1 scale
- **Generalization Index**: Quantitative measure of multi-task capability
- **Difficulty Index**: Task complexity assessment metric

## Generated Visualizations

All publication-ready figures are automatically saved to the `plots/` directory in multiple formats (PNG/SVG/PDF):

- **Forest Plot**: Performance comparison across VLA models with confidence intervals
- **Encoder Analysis**: 4-panel analysis of visual encoder impact on model performance
- **Domain Component Analysis**: Cross-domain performance patterns and transfer learning effectiveness  
- **Fusion Theory Visualization**: 3-panel theoretical framework visualization
- **VLA-FEB Histogram**: Distribution analysis of composite evaluation scores
- **Scale Analysis**: Model scale vs. performance relationship analysis


# VLA Models

![VLA Models Trend](https://github.com/Muhayyuddin/VLAs/blob/main/VLA.png)
The top row presents major VLA
models introduced each year, alongside their associated institutions. The bottom row
displays key datasets used to train and evaluate VLA models, grouped by release year. The figure highlights the
increasing scale and diversity of datasets and institutional involvement, with contributions from academic (e.g.,
CMU, CNRS, UC, Peking Uni) and industrial labs (e.g., Google, NVIDIA, Microsoft). This timeline highlights
the rapid advancements in VLA research.

Below is the list of the VLAs reviewed in the paper

[2022][Cliport: What and where pathways for robotic manipulation](https://proceedings.mlr.press/v164/shridhar22a/shridhar22a.pdf)  
[2022][Rt-1: Robotics transformer for real‑world control at scale](https://arxiv.org/abs/2212.06817)  
[2022][A Generalist Agent](https://arxiv.org/abs/2205.06175)  
[2022][VIMA: General Robot Manipulation with Multimodal Prompts](https://arxiv.org/abs/2210.03094)  
[2022][PERCEIVER-ACTOR:A Multi-Task Transformer for Robotic Manipulation](https://peract.github.io/paper/peract_corl2022.pdf) <br>
[2022][Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)  
[2023][RoboAgent: Generalist Robot Agent with Semantic and Temporal Understanding](https://arxiv.org/abs/2310.08560)  
[2023][Robotic Task Generalization via Hindsight Trajectory Sketches](https://arxiv.org/abs/2311.01977)  
[2023][Learning fine‑grained bimanual manipulation with low‑cost hardware](https://arxiv.org/abs/2304.13705)  
[2023][Rt-2: Vision‑language‑action models transfer web knowledge to robotic control](Link TBD)  
[2023][Voxposer: Composable 3D value maps for robotic manipulation with language models](https://arxiv.org/abs/2307.05973)  
[2024][CLIP‑RT: Learning Language‑Conditioned Robotic Policies with Natural Language Supervision](https://arxiv.org/abs/2411.00508)  
[2023][Diffusion Policy: Visuomotor policy learning via action diffusion](https://arxiv.org/pdf/2303.04137)  
[2024][Octo: An open‑source generalist robot policy](https://arxiv.org/abs/2405.12213)  
[2024][Towards testing and evaluating vision‑language manipulation: An empirical study](https://arxiv.org/abs/2409.12894)  
[2024][NaVILA: Legged robot vision‑language‑action model for navigation](https://arxiv.org/abs/2412.04453)  
[2024][RoboNurse‑VLA: Real‑time voice‑to‑action pipeline for surgical instrument handover](https://arxiv.org/pdf/2409.19590)  
[2024][Mobility VLA: Multimodal instruction navigation with topological mapping](https://arxiv.org/pdf/2407.07775)  
[2024][ReVLA: Domain adaptation adapters for robotic foundation models](https://arxiv.org/pdf/2409.15250.pdf)  
[2024][Uni‑NaVid: Video‑based VLA unifying embodied navigation tasks](https://arxiv.org/pdf/2412.06224.pdf)  
[2024][RDT‑1B: 1.2B‑parameter diffusion foundation model for manipulation](https://arxiv.org/pdf/2410.07864.pdf)  
[2024][RoboMamba: Mamba‑based unified VLA with linear‑time inference](https://arxiv.org/pdf/2406.04339.pdf)   
[2024][Chain‑of‑Affordance: Sequential affordance reasoning for spatial planning](https://arxiv.org/pdf/2412.20451.pdf)  
[2024][Edge VLA:Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities](https://arxiv.org/pdf/2403.04908)  
[2024][OpenVLA: LORA‑fine‑tuned open‑source VLA with high‑success transfer](https://arxiv.org/pdf/2406.09246.pdf)  
[2024][CogACT: Componentized diffusion action transformer for VLA](https://arxiv.org/pdf/2411.19650.pdf)  
[2024][ShowUI‑2B: GUI/web navigation via screenshot grounding and token selection](https://arxiv.org/pdf/2411.17465)  
[2024][HiRT: Hierarchical planning/control separation for VLA](https://arxiv.org/pdf/2410.05273)  
[2024][Pi‑0: General robot control flow model for open‑world tasks](https://arxiv.org/pdf/2410.24164.pdf) <br> 
[2024][A3VLM: Articulation‑aware affordance grounding from RGB video](https://arxiv.org/pdf/2406.07549.pdf)  
[2024][SVLR: Modular “segment‑to‑action” pipeline using visual prompt retrieval](https://arxiv.org/pdf/2502.01071.pdf)  
[2024][Bi‑VLA: Dual‑arm instruction‑to‑action planner for recipe demonstrations](https://arxiv.org/pdf/2405.06039.pdf)  
[2024][QUAR‑VLA: Quadruped‑specific VLA with adaptive gait mapping](https://arxiv.org/pdf/2312.14457.pdf)  
[2024][3D‑VLA: Integrating 3D generative diffusion heads for world reconstruction](https://arxiv.org/pdf/2403.09631)  
[2024][RoboMM: MIM‑based multimodal decoder unifying 3D perception and language](https://arxiv.org/pdf/2412.07215.pdf)  
[2025][FAST: Frequency‑space action tokenization for faster inference](https://arxiv.org/pdf/2501.09747.pdf)  
[2025][OpenVLA‑OFT: Optimized fine‑tuning of OpenVLA with parallel decoding](https://arxiv.org/pdf/2502.19645.pdf)  
[2025][CoVLA: Autonomous driving VLA trained on annotated scene data](https://arxiv.org/pdf/2408.10845.pdf)  
[2025][ORION: Holistic end‑to‑end driving VLA with semantic trajectory control](https://arxiv.org/pdf/2503.19755.pdf)  
[2025][UAV‑VLA: Zero‑shot aerial mission VLA combining satellite/UAV imagery](https://arxiv.org/pdf/2501.05014.pdf)  
[2025][Combat VLA: Ultra‑fast tactical reasoning in 3D environments](https://arxiv.org/pdf/2503.09527.pdf)  
[2025][HybridVLA: Ensemble decoding combining diffusion and autoregressive policies](https://arxiv.org/pdf/2503.10631.pdf)  
[2025][NORA: Low‑overhead VLA with integrated visual reasoning and FAST decoding](https://arxiv.org/pdf/2504.19854.pdf)  
[2025][SpatialVLA: 3D spatial encoding and adaptive action discretization](https://arxiv.org/pdf/2501.15830.pdf)  
[2025][MoLe‑VLA: Selective layer activation for faster inference](https://arxiv.org/pdf/2503.20384.pdf)  
[2025][JARVIS‑VLA: Open‑world instruction following in 3D games with keyboard/mouse](https://arxiv.org/pdf/2503.16365.pdf)  
[2025][UP‑VLA: Unified understanding and prediction model for embodied agents](https://arxiv.org/pdf/2501.18867.pdf)  
[2025][Shake‑VLA: Modular bimanual VLA for cocktail‑mixing tasks](https://arxiv.org/pdf/2501.06919.pdf)  
[2025][MORE: Scalable mixture‑of‑experts RL for VLA models](https://arxiv.org/pdf/2503.08007.pdf)  
[2025][DexGraspVLA: Diffusion‑based dexterous grasping framework](https://arxiv.org/pdf/2502.20900.pdf)  
[2025][DexVLA: Cross‑embodiment diffusion expert for rapid adaptation](https://arxiv.org/pdf/2502.05855.pdf)  
[2025][Humanoid‑VLA: Hierarchical full‑body humanoid control VLA](https://arxiv.org/pdf/2502.14795.pdf)  
[2025][ObjectVLA: End‑to‑end open‑world object manipulation](https://arxiv.org/pdf/2502.19250.pdf)  
[2025][Gemini Robotics: Bringing AI into the Physical World](https://arxiv.org/pdf/2503.20020.pdf)  
[2025][ECoT: Robotic Control via Embodied Chain‑of‑Thought Reasoning](https://arxiv.org/pdf/2407.08693.pdf)  
[2025][OTTER: A Vision‑Language‑Action Model with Text‑Aware Visual Feature Extraction](https://arxiv.org/pdf/2503.03734.pdf)  
[2025][π‑0.5: A VLA Model with Open‑World Generalization](https://arxiv.org/pdf/2504.16054.pdf)  
[2025][OneTwoVLA: A Unified Model with Adaptive Reasoning](https://arxiv.org/pdf/2505.11917.pdf)  
[2025][Helix: A Vision-Language-Action Model for Generalist Humanoid Control](https://www.figure.ai/news/helix)<br>
[2025][SmolVLA: A Vision‑Language‑Action Model for Affordable and Efficient Robotics](https://arxiv.org/pdf/2506.01844.pdf)  
[2025][EF‑VLA: Vision‑Language‑Action Early Fusion with Causal Transformers](https://openreview.net/pdf/32c153a3b16174884cf62b285adbfbdcc57b163e.pdf)  
[2025][PD‑VLA: Accelerating vision‑language‑action inference via parallel decoding](https://arxiv.org/pdf/2503.02310.pdf)  
[2025][LeVERB: Humanoid Whole‑Body Control via Latent Verb Generation](https://arxiv.org/pdf/2506.13751.pdf)  
[2025][TLA: Tactile‑Language‑Action Model for High‑Precision Contact Tasks](https://arxiv.org/pdf/2503.08548.pdf)  
[2025][Interleave‑VLA: Enhancing VLM‑LLM interleaved instruction processing](https://arxiv.org/pdf/2505.02152.pdf)  
[2025][iRe‑VLA: Iterative reinforcement and supervised fine‑tuning for robust VLA](https://arxiv.org/pdf/2501.16664.pdf)  
[2025][TraceVLA: Visual trace prompting for spatio‑temporal manipulation cues](https://arxiv.org/pdf/2412.10345.pdf)  
[2025][OpenDrive VLA: End‑to‑End Driving with Semantic Scene Alignment](https://arxiv.org/pdf/2503.23463.pdf)  
[2025][V‑JEPA 2: Dual‑Stream Video JEPA for Predictive Robotic Planning](https://arxiv.org/pdf/2506.09985.pdf)  
[2025][Knowledge Insulating VLA: Insulation Layers for Modular VLA Training](https://arxiv.org/pdf/2505.23705.pdf)  
[2025][GR00T N1: Diffusion Foundation Model for Humanoid Control](https://arxiv.org/pdf/2503.14734.pdf)  
[2025][AgiBot World Colosseo: Unified Embodied Dataset Platform](https://arxiv.org/pdf/2503.06669.pdf)  
[2025][Hi Robot: Hierarchical Planning and Control for Complex Environments](https://arxiv.org/pdf/2502.19417.pdf)  
[2025][EnerVerse: World‑Model LLM for Long‑Horizon Manipulation](https://arxiv.org/pdf/2501.01895.pdf)  
[2024][FLaRe: Large-Scale RL Fine-Tuning for Adaptive Robotic Policies](https://arxiv.org/pdf/2409.16578.pdf)  
[2025][Beyond Sight: Sensor Fusion via Language-Grounded Attention](https://arxiv.org/pdf/2501.04693.pdf)  
[2025][GeoManip: Geometric Constraint Encoding for Robust Manipulation](https://arxiv.org/pdf/2501.09783.pdf)  
[2025][Universal Actions: Standardizing Action Dictionaries for Transfer](https://arxiv.org/pdf/2501.10105.pdf)  
[2025][RoboHorizon: Multi-View Environment Modeling with LLM Planning](https://arxiv.org/pdf/2501.06605.pdf)  
[2025][SAM2Act: Segmentation‑Augmented Memory for Object‑Centric Manipulation](https://arxiv.org/pdf/2501.18564.pdf)  
[2025][VLA‑Cache: Token Caching for Efficient VLA Inference](https://arxiv.org/pdf/2502.02175.pdf)  
[2025][Forethought VLA: Latent Alignment for Foresight‑Driven Policies](https://arxiv.org/pdf/2502.01828.pdf)  
[2024][GRAPE: Preference‑Guided Policy Adaptation via Feedback](https://arxiv.org/pdf/2409.16578.pdf)  
[2025][HAMSTER: Hierarchical Skill Decomposition for Multi‑Step Manipulation](https://arxiv.org/pdf/2502.05485.pdf)  
[2025][TempoRep VLA: Successor Representation for Temporal Planning](https://arxiv.org/pdf/2507.10672v1)  
[2025][ConRFT: Consistency Regularized Fine‑Tuning with Reinforcement](https://arxiv.org/pdf/2502.05450.pdf)  
[2025][RoboBERT: Unified Multimodal Transformer for Manipulation](https://arxiv.org/pdf/2502.07837.pdf)  
[2024][Diffusion Transformer Policy: Robust Multimodal Action Sampling](https://arxiv.org/pdf/2410.15959.pdf)  
[2025][GEVRM: Generative Video Modeling for Goal‑Oriented Planning](https://arxiv.org/pdf/2502.09268.pdf)  
[2025][SoFar: Successor‑Feature Orientation Representations](https://arxiv.org/pdf/2502.13143.pdf)  
[2025][ARM4R: Auto‑Regressive 4D Transition Modeling for Trajectories](https://arxiv.org/pdf/2502.13142.pdf)  
[2025][Magma: Foundation Multimodal Agent Model for Control](https://arxiv.org/pdf/2502.13130.pdf)  
[2025][An Atomic Skill Library: Modular Skill Composition for Robotics](https://arxiv.org/pdf/2501.15068.pdf)  
[2025][RoboBrain: Knowledge‑Grounded Policy Brain for Multimodal Tasks](https://arxiv.org/pdf/2502.21257.pdf)  
[2025][SafeVLA: Safety‑Aware Vision‑Language‑Action Policies](https://arxiv.org/pdf/2503.03480.pdf)  
[2025][CognitiveDrone: Embodied Reasoning VLA for UAV Planning](https://arxiv.org/pdf/2503.01378.pdf)  
[2025][VLAS: Voice‑Driven Vision‑Language‑Action Control](https://arxiv.org/pdf/2502.13508.pdf)  
[2025][ChatVLA: Conversational VLA for Interactive Control](https://arxiv.org/pdf/2502.14420.pdf)  
[2024][Diffusion‑VLA: Diffusion‑Based Policy for Generalizable Manipulation](https://arxiv.org/pdf/2412.03293.pdf)  
[2025][RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics](https://arxiv.org/pdf/2506.04308.pdf)  

# Datasets
[2018][EmbodiedQA: Embodied Question Answering](https://openaccess.thecvf.com/content_cvpr_2018/papers_backup/Das_Embodied_Question_Answering_CVPR_2018_paper.pdf)  
[2018][R2R: Vision‑and‑Language Navigation: Interpreting Visually‑Grounded Navigation Instructions in Real Environments](https://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Vision-and-Language_Navigation_Interpreting_CVPR_2018_paper.pdf)  
[2020][ALFRED](https://arxiv.org/abs/1912.01734)  
[2020][RLBench: The Robot Learning Benchmark & Learning Environment](https://arxiv.org/pdf/1909.12271.pdf)  
[2019][Vision‑and‑Dialog Navigation](https://arxiv.org/abs/1907.04957)  
[2021][TEACh: Task‑driven Embodied Agents that Chat](https://arxiv.org/abs/2110.00534)  
[2022][DialFRED: Dialogue‑Enabled Agents for Embodied Instruction Following](https://arxiv.org/pdf/2202.13330.pdf)  
[2022][Ego4D: Around the World in 3,000 Hours of Egocentric Video](https://arxiv.org/abs/2110.07058)  
[2022][CALVIN: A Benchmark for Language‑Conditioned Long‑Horizon Robot Manipulation Tasks](https://arxiv.org/abs/2112.03227)  
[2024][DROID: A Large‑Scale In‑The‑Wild Robot Manipulation Dataset](https://droid-dataset.github.io/)  
[2025][Open X-Embodiment: Robotic Learning Datasets and RT‑X Models](https://arxiv.org/abs/2310.08864)  
[2025][RoboSpatial: Teaching Spatial Understanding via Vision‑Language Models for Robotics](https://arxiv.org/abs/2411.16537)  
[2024][CoVLA: Comprehensive Vision‑Language‑Action Dataset for Autonomous Driving](https://arxiv.org/abs/2408.10845)  
[2025][TLA: Tactile‑Language‑Action Model for Contact‑Rich Manipulation](https://arxiv.org/abs/2503.08548)  
[2023][BridgeData V2: A Dataset for Robot Learning at Scale](https://proceedings.mlr.press/v229/walke23a/walke23a.pdf)  
[2023][LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://proceedings.neurips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf)  
[2025][Kaiwu: A Multimodal Manipulation Dataset and Framework for Robotic Perception and Interaction](https://arxiv.org/abs/2503.05231)  
[2025][PLAICraft: Large‑Scale Time‑Aligned Vision‑Speech‑Action Dataset for Embodied AI](https://arxiv.org/abs/2505.12707)  
[2025][AgiBot World Colosseo: A Large‑Scale Manipulation Dataset for Intelligent Embodied Systems](https://arxiv.org/abs/2503.06669)  
[2023][Robo360: A 3D Omnispective Multi‑Modal Robotic Manipulation Dataset](https://arxiv.org/abs/2312.06686)  
[2025][REASSEMBLE: A Multimodal Dataset for Contact‑Rich Robotic Assembly and Disassembly](https://arxiv.org/abs/2502.05086)  
[2025][RoboCerebra: A Large‑Scale Benchmark for Long‑Horizon Robotic Manipulation Evaluation](https://arxiv.org/abs/2506.06677)  
[2025][IRef‑VLA: A Benchmark for Interactive Referential Grounding with Imperfect Language in 3D Scenes](https://arxiv.org/abs/2503.17406)  
[2025][Interleave‑VLA: Enhancing Robot Manipulation with Interleaved Image‑Text Instructions](https://arxiv.org/abs/2406.07000)  
[2024][RoboMM: All‑in‑One Multimodal Large Model for Robotic Manipulation](https://arxiv.org/abs/2412.07215)  
[2024][All Robots in One: A New Standard and Unified Dataset for Versatile, General‑Purpose Embodied Agents](https://arxiv.org/abs/2408.10899)
[2025][RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics](https://arxiv.org/pdf/2506.04308.pdf)  

# Simulators
[2017][AI2-THOR:][AI2-THOR](https://ai2thor.allenai.org)  
[2019][Habitat:][Habitat](https://aihabitat.org)  
[2020][NVIDIA Isaac Sim:][NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim)  
[2004][Gazebo:][Gazebo](http://gazebosim.org)  
[2016][PyBullet:][PyBullet](https://pybullet.org)  
[2013][CoppeliaSim:][CoppeliaSim](https://www.coppeliarobotics.com)  
[2004][Webots:][Webots](https://cyberbotics.com)  
[2018][Unity ML‑Agents:][Unity ML‑Agents](https://unity-technologies.github.io/ml-agents/)  
[2012][MuJoCo:][MuJoCo](https://mujoco.org)  
[2020][iGibson:][iGibson](https://svl.stanford.edu/igibson)  
[2023][UniSim:][UniSim](https://universal-simulator.github.io/unisim/)  
[2020][SAPIEN:][SAPIEN](https://sapien.ucsd.edu) 



