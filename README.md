# Vision–Language–Action (VLA) Models in Robotics

This repository was developed alongside the paper **“Vision–Language–Action (VLA) Models in Robotics”** and provides a living catalog of:

- **VLA Models**  
  Key vision–language–action architectures, with publication year, core capabilities, training datasets, robot platforms, summary descriptions, and links to the original papers and project websites.

- **Datasets**  
  Major benchmarks and large‑scale collections used to train and evaluate VLA systems, including QA/navigation corpora, manipulation demonstrations, and multimodal embodiment data.

- **Simulators**  
  Widely adopted simulation platforms for generating VLA data—spanning photorealistic navigation, dexterous manipulation, multi‑robot coordination, and more—each linked to its official website.

We aim to keep this list up to date as new VLA models, datasets, and simulation tools emerge. Contributions and pull requests adding recently published work or tooling are most welcome!  

![Dataset Benchmarking](https://github.com/Muhayyuddin/VLAs/blob/main/benchmarkdataset.png)

[Code](https://github.com/Muhayyuddin/VLAs/blob/main/dataset_plot.py)

---

## Table of Contents
- [VLA Models](#vla-models)  
- [Datasets](#datasets)  
- [Simulators](#simulators)



---
# VLA Models

[2022][CLIPort:][Cliport: What and where pathways for robotic manipulation](https://proceedings.mlr.press/v164/shridhar22a/shridhar22a.pdf)  
[2022][RT-1:][Rt-1: Robotics transformer for real‑world control at scale](https://arxiv.org/abs/2212.06817)  
[2022][Gato:][A Generalist Agent](https://arxiv.org/abs/2205.06175)  
[2022][VIMA:][VIMA: General Robot Manipulation with Multimodal Prompts](https://arxiv.org/abs/2210.03094)  
[2022][PerAct:][PERCEIVER-ACTOR:A Multi-Task Transformer for Robotic Manipulation](https://peract.github.io/paper/peract_corl2022.pdf) <br>
[2022][SayCan:][Do As I Can, Not As I Say: Grounding Language in Robotic Affordances](https://arxiv.org/abs/2204.01691)  
[2023][RoboAgent:][RoboAgent: Generalist Robot Agent with Semantic and Temporal Understanding](https://arxiv.org/abs/2310.08560)  
[2023][RT-Trajectory:][Robotic Task Generalization via Hindsight Trajectory Sketches](https://arxiv.org/abs/2311.01977)  
[2023][ACT:][Learning fine‑grained bimanual manipulation with low‑cost hardware](https://arxiv.org/abs/2304.13705)  
[2023][RT-2:][Rt-2: Vision‑language‑action models transfer web knowledge to robotic control](Link TBD)  
[2023][VoxPoser:][Voxposer: Composable 3D value maps for robotic manipulation with language models](https://arxiv.org/abs/2307.05973)  
[2024][CLIP-RT:][CLIP‑RT: Learning Language‑Conditioned Robotic Policies with Natural Language Supervision](https://arxiv.org/abs/2411.00508)  
[2023][Diffusion Policy:][Diffusion Policy: Visuomotor policy learning via action diffusion](https://arxiv.org/pdf/2303.04137)  
[2024][Octo:][Octo: An open‑source generalist robot policy](https://arxiv.org/abs/2405.12213)  
[2024][VLATest:][Towards testing and evaluating vision‑language manipulation: An empirical study](https://arxiv.org/abs/2409.12894)  
[2024][NaVILA:][NaVILA: Legged robot vision‑language‑action model for navigation](https://arxiv.org/abs/2412.04453)  
[2024][RoboNurse‑VLA:][RoboNurse‑VLA: Real‑time voice‑to‑action pipeline for surgical instrument handover](https://arxiv.org/pdf/2409.19590)  
[2024][Mobility VLA:][Mobility VLA: Multimodal instruction navigation with topological mapping](https://arxiv.org/pdf/2407.07775)  
[2024][RevLA:][RevLA: Domain adaptation adapters for robotic foundation models](Link TBD)  
[2024][Uni-NaVid:][Uni-NaVid: Video‑based VLA unifying embodied navigation tasks](Link TBD)  
[2024][RDT-1B:][RDT‑1B: 1.2B‑parameter diffusion foundation model for manipulation](Link TBD)  
[2024][RoboMamba:][RoboMamba: Mamba‑based unified VLA with linear‑time inference](Link TBD)  
[2024][Chain‑of‑Affordance:][Chain‑of‑Affordance: Sequential affordance reasoning for spatial planning](Link TBD)  
[2024][Edge VLA:][Edge VLA: Lightweight, edge‑optimized VLA for low‑power real‑time inference](Link TBD)  
[2024][OpenVLA:][OpenVLA: LORA‑fine‑tuned open‑source VLA with high-success transfer](Link TBD)  
[2024][CogACT:][CogACT: Componentized diffusion action transformer for VLA](Link TBD)  
[2024][ShowUI‑2B:][ShowUI‑2B: GUI/web navigation via screenshot grounding and token selection](Link TBD)  
[2024][Pi‑0:][Pi‑0: General robot control flow model for open‑world tasks](Link TBD)  
[2024][HiRT:][HiRT: Hierarchical planning/control separation for VLA](Link TBD)  
[2024][A3VLM:][A3VLM: Articulation‑aware affordance grounding from RGB video](Link TBD)  
[2024][SVLR:][SVLR: Modular “segment‑to‑action” pipeline using visual prompt retrieval](Link TBD)  
[2024][Bi‑VLA:][Bi‑VLA: Dual‑arm instruction‑to‑action planner for recipe demonstrations](Link TBD)  
[2024][QUAR‑VLA:][QUAR‑VLA: Quadruped‑specific VLA with adaptive gait mapping](Link TBD)  
[2024][3D‑VLA:][3D‑VLA: Integrating 3D generative diffusion heads for world reconstruction](Link TBD)  
[2024][RoboMM:][RoboMM: MIM‑based multimodal decoder unifying 3D perception and language](Link TBD)  
[2025][FAST:][FAST: Frequency‑space action tokenization for faster inference](Link TBD)  
[2025][OpenVLA‑OFT:][OpenVLA‑OFT: Optimized fine‑tuning of OpenVLA with parallel decoding](Link TBD)  
[2025][CoVLA:][CoVLA: Autonomous driving VLA trained on annotated scene data](Link TBD)  
[2025][ORION:][ORION: Holistic end‑to‑end driving VLA with semantic trajectory control](Link TBD)  
[2025][UAV‑VLA:][UAV‑VLA: Zero‑shot aerial mission VLA combining satellite/UAV imagery](Link TBD)  
[2025][Combat VLA:][Combat VLA: Ultra‑fast tactical reasoning in 3D environments](Link TBD)  
[2025][HybridVLA:][HybridVLA: Ensemble decoding combining diffusion and autoregressive policies](Link TBD)  
[2025][NORA:][NORA: Low‑overhead VLA with integrated visual reasoning and FAST decoding](Link TBD)  
[2025][SpatialVLA:][SpatialVLA: 3D spatial encoding and adaptive action discretization](Link TBD)  
[2025][MoLe‑VLA:][MoLe‑VLA: Selective layer activation for faster inference](Link TBD)  
[2025][JARVIS‑VLA:][JARVIS‑VLA: Open‑world instruction following in 3D games with keyboard/mouse](Link TBD)  
[2025][UP‑VLA:][UP‑VLA: Unified understanding and prediction model for embodied agents](Link TBD)  
[2025][Shake‑VLA:][Shake‑VLA: Modular bimanual VLA for cocktail‑mixing tasks](Link TBD)  
[2025][MORE:][MORE: Scalable mixture-of-experts RL for VLA models](Link TBD)  
[2025][DexGraspVLA:][DexGraspVLA: Diffusion‑based dexterous grasping framework](Link TBD)  
[2025][DexVLA:][DexVLA: Cross‑embodiment diffusion expert for rapid adaptation](Link TBD)  
[2025][Humanoid‑VLA:][Humanoid‑VLA: Hierarchical full‑body humanoid control VLA](Link TBD)  
[2025][ObjectVLA:][ObjectVLA: End‑to‑end open‑world object manipulation](Link TBD)  
[2025][Gemini Robotics:][Gemini Robotics: Bringing AI into the Physical World](Link TBD)  
[2025][ECoT:][Robotic Control via Embodied Chain‑of‑Thought Reasoning](Link TBD)  
[2025][OTTER:][OTTER: A Vision‑Language‑Action Model with Text‑Aware Visual Feature Extraction](Link TBD)  
[2025][pi‑0.5:][π‑0.5: A VLA Model with Open‑World Generalization](Link TBD)  
[2025][OneTwoVLA:][OneTwoVLA: A Unified Model with Adaptive Reasoning](Link TBD)  
[2025][Helix:][Helix: A Vision-Language-Action Model for Generalist Humanoid Control](https://www.figure.ai/news/helix)<br>
[2025][SmolVLA:][SmolVLA: A Vision‑Language‑Action Model for Affordable and Efficient Robotics](Link TBD) <br>
[2025][EF-VLA:][EF-VLA: Vision-Language-Action Early Fusion with Causal Transformers](Link TBD)  
[2025][PD-VLA:][Accelerating vision-language-action inference via parallel decoding](Link TBD)  
[2025][LeVERB:][LeVERB: Humanoid Whole-Body Control via Latent Verb Generation](Link TBD)  
[2025][TLA:][TLA: Tactile-Language-Action Model for High-Precision Contact Tasks](Link TBD)  
[2025][Interleave-VLA:][Interleave-VLA: Enhancing VLM-LLM Interleaved Instruction Processing](Link TBD)  
[2025][iRe-VLA:][iRe-VLA: Iterative Reinforcement and Supervised Fine-Tuning for Robust VLA](Link TBD)  
[2025][TraceVLA:][TraceVLA: Visual Trace Prompting for Spatio-Temporal Manipulation Cues](Link TBD)  
[2025][OpenDrive VLA:][OpenDrive VLA: End-to-End Driving with Semantic Scene Alignment](Link TBD)  
[2025][V-JEPA 2:][V-JEPA 2: Dual-Stream Video JEPA for Predictive Robotic Planning](Link TBD)  
[2025][Knowledge Insulating VLA:][Knowledge Insulating VLA: Insulation Layers for Modular VLA Training](Link TBD)  
[2025][GR00T N1:][GR00T N1: Diffusion Foundation Model for Humanoid Control](Link TBD)  
[2025][AgiBot World Colosseo:][AgiBot World Colosseo: Unified Embodied Dataset Platform](Link TBD)  
[2025][Hi Robot:][Hi Robot: Hierarchical Planning and Control for Complex Environments](Link TBD)  
[2025][EnerVerse:][EnerVerse: World-Model LLM for Long-Horizon Manipulation](Link TBD)  
[2024][FLaRe:][FLaRe: Large-Scale RL Fine-Tuning for Adaptive Robotic Policies](Link TBD)  
[2025][Beyond Sight:][Beyond Sight: Sensor Fusion via Language-Grounded Attention](Link TBD)  
[2025][GeoManip:][GeoManip: Geometric Constraint Encoding for Robust Manipulation](Link TBD)  
[2025][Universal Actions:][Universal Actions: Standardizing Action Dictionaries for Transfer](Link TBD)  
[2025][RoboHorizon:][RoboHorizon: Multi-View Environment Modeling with LLM Planning](Link TBD)  
[2025][SAM2Act:][SAM2Act: Segmentation-Augmented Memory for Object-Centric Manipulation](Link TBD)  
[2025][LMM Planner Integration:][LMM Planner Integration: Merging 3D Policies with Strategic LMM Planning](Link TBD)  
[2025][VLA-Cache:][VLA-Cache: Token Caching for Efficient VLA Inference](Link TBD)  
[2025][Forethought VLA:][Forethought VLA: Latent Alignment for Foresight-Driven Policies](Link TBD)  
[2024][GRAPE:][GRAPE: Preference-Guided Policy Adaptation via Feedback](Link TBD)  
[2025][HAMSTER:][HAMSTER: Hierarchical Skill Decomposition for Multi-Step Manipulation](Link TBD)  
[2025][TempoRep VLA:][TempoRep VLA: Successor Representation for Temporal Planning](Link TBD)  
[2025][ConRFT:][ConRFT: Consistency Regularized Fine-Tuning with Reinforcement](Link TBD)  
[2025][RoboBERT:][RoboBERT: Unified Multimodal Transformer for Manipulation](Link TBD)  
[2024][Diffusion Transformer Policy:][Diffusion Transformer Policy: Robust Multimodal Action Sampling](Link TBD)  
[2025][GEVRM:][GEVRM: Generative Video Modeling for Goal-Oriented Planning](Link TBD)  
[2025][SoFar:][SoFar: Successor-Feature Orientation Representations](Link TBD)  
[2025][ARM4R:][ARM4R: Auto-Regressive 4D Transition Modeling for Trajectories](Link TBD)  
[2025][Magma:][Magma: Foundation Multimodal Agent Model for Control](Link TBD)  
[2025][An Atomic Skill Library:][An Atomic Skill Library: Modular Skill Composition for Robotics](Link TBD)  
[2025][VLAS:][VLAS: Voice-Driven Vision-Language-Action Control](Link TBD)  
[2025][ChatVLA:][ChatVLA: Conversational VLA for Interactive Control](Link TBD)  
[2025][RoboBrain:][RoboBrain: Knowledge-Grounded Policy Brain for Multimodal Tasks](Link TBD)  
[2025][SafeVLA:][SafeVLA: Safety-Aware Vision-Language-Action Policies](Link TBD)  
[2025][CognitiveDrone:][CognitiveDrone: Embodied Reasoning VLA for UAV Planning](Link TBD)  
[2024][Diffusion-VLA:][Diffusion-VLA: Diffusion-Based Policy for Generalizable Manipulation](Link TBD)  

# Datasets
[2018][EmbodiedQA:][Embodied Question Answering](Link TBD)  
[2018][R2R:][Vision-and-Language Navigation: Interpreting Visually‑Grounded Navigation Instructions in Real Environments](Link TBD)  
[2020][ALFRED:][ALFRED](https://arxiv.org/abs/1912.01734)  
[2020][RLBench:][RLBench: The Robot Learning Benchmark & Learning Environment](Link TBD)  
[2019][CVDN (NDH):][Vision‑and‑Dialog Navigation](https://arxiv.org/abs/1907.04957)  
[2021][TEACh:][TEACh: Task‑driven Embodied Agents that Chat](https://arxiv.org/abs/2110.00534)  
[2022][DialFRED:][DialFRED: Dialogue‑Enabled Agents for Embodied Instruction Following](Link TBD)  
[2022][Ego4D:][Ego4D: Around the World in 3,000 Hours of Egocentric Video](https://arxiv.org/abs/2110.07058)  
[2022][CALVIN:][CALVIN: A Benchmark for Language‑Conditioned Long‑Horizon Robot Manipulation Tasks](https://arxiv.org/abs/2112.03227)  
[2024][DROID:][DROID: A Large‑Scale In‑The‑Wild Robot Manipulation Dataset](https://droid-dataset.github.io/)  
[2025][Open X-Embodiment:][Open X-Embodiment: Robotic Learning Datasets and RT‑X Models](https://arxiv.org/abs/2310.08864)  
[2025][RoboSpatial:][RoboSpatial: Teaching Spatial Understanding via Vision‑Language Models for Robotics](https://arxiv.org/abs/2411.16537)  
[2024][CoVLA:][CoVLA: Comprehensive Vision‑Language‑Action Dataset for Autonomous Driving](https://arxiv.org/abs/2408.10845)  
[2025][TLA:][TLA: Tactile‑Language‑Action Model for Contact‑Rich Manipulation](https://arxiv.org/abs/2503.08548)  
[2023][BridgeData V2:][BridgeData V2: A Dataset for Robot Learning at Scale](Link TBD)  
[2023][LIBERO:][LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](Link TBD)  
[2025][Kaiwu:][Kaiwu: A Multimodal Manipulation Dataset and Framework for Robotic Perception and Interaction](https://arxiv.org/abs/2503.05231)  
[2025][PLAICraft:][PLAICraft: Large‑Scale Time‑Aligned Vision‑Speech‑Action Dataset for Embodied AI](https://arxiv.org/abs/2505.12707)  
[2025][AgiBot World:][AgiBot World Colosseo: A Large‑Scale Manipulation Dataset for Intelligent Embodied Systems](https://arxiv.org/abs/2503.06669)  
[2023][Robo360:][Robo360: A 3D Omnispective Multi‑Modal Robotic Manipulation Dataset](https://arxiv.org/abs/2312.06686)  
[2025][REASSEMBLE:][REASSEMBLE: A Multimodal Dataset for Contact‑Rich Robotic Assembly and Disassembly](https://arxiv.org/abs/2502.05086)  
[2025][RoboCerebra:][RoboCerebra: A Large‑Scale Benchmark for Long‑Horizon Robotic Manipulation Evaluation](https://arxiv.org/abs/2506.06677)  
[2025][IRef‑VLA:][IRef‑VLA: A Benchmark for Interactive Referential Grounding with Imperfect Language in 3D Scenes](https://arxiv.org/abs/2503.17406)  
[2025][Interleave‑VLA:][Interleave‑VLA: Enhancing Robot Manipulation with Interleaved Image‑Text Instructions](https://arxiv.org/abs/2406.07000)  
[2024][RoboMM:][RoboMM: All‑in‑One Multimodal Large Model for Robotic Manipulation](https://arxiv.org/abs/2412.07215)  
[2024][ARIO:][All Robots in One: A New Standard and Unified Dataset for Versatile, General‑Purpose Embodied Agents](https://arxiv.org/abs/2408.10899)

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




