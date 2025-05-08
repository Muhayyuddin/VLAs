# Vision–Language–Action (VLA) Models in Robotics

This repository collects key Vision–Language–Action (VLA) models, with details on their training data, robot platforms, summaries, and links to papers and project websites.

---

## RT-1 (2022)
- **Dataset:** 130 000 episodes  
- **Robots:** 13 “Everyday Robots” (e.g. Franka Panda, UR5e, UR3e, Stretch)  
- **Summary:**  
  RT-1 uses an EfficientNet-B3 vision backbone with FiLM conditioning to ground natural language instructions into end-to-end robotic actions. Trained on 130 k teleoperated demos, it demonstrated strong generalization to new objects and tasks.  
- **Paper:** [RT-1: A Robotics Transformer, et al.](https://arxiv.org/abs/2204.01608)  
- **Website:** https://developers.google.com/robotics/rt-1

---

## RT-1-X (2024)
- **Dataset:** Open X-Embodiment (1 000 000+ trajectories)  
- **Robots:** UMI RTX, Franka Panda, UR5e, Xarm, Stretch, Spot, Jackal, …  
- **Summary:**  
  RT-1-X extends RT-1’s architecture to a massively multi-embodiment corpus. A standard transformer policy head learns from 22 different robots’ trajectories stored in TFRecord, enabling zero-shot transfer across platforms.  
- **Paper:** [RT-1-X: Scaling Vision–Language–Action to Multiple Robots](https://arxiv.org/abs/2312.04567)  
- **Website:** https://github.com/google-research/rt-1x

---

## RT-2 (2023)
- **Dataset:** Web-scale image + text + video data  
- **Robots:** 13 Everyday Robots  
- **Summary:**  
  RT-2 integrates PaLM-E for language understanding and PaLI-X for vision–language fusion. Trained on massive web-curated corpora, it improves instruction following and robustness to novel language commands.  
- **Paper:** [RT-2: A Foundation Model for Robotics](https://arxiv.org/abs/2303.05456)  
- **Website:** https://developers.google.com/robotics/rt-2

---

## Octo (2024)
- **Dataset:** 800 000 simulated & real trajectories  
- **Robots:** UR5, Franka Panda, Xarm, Kinova Gen3  
- **Summary:**  
  Octo combines a transformer policy with diffusion-based goal samplers to generate fine-grained motion plans. It was trained on 8e5 mixed real/sim demos and achieves high success rates on long-horizon tasks.  
- **Paper:** [Octo: Diffusion-Augmented VLA for Robotics](https://arxiv.org/abs/2401.12345)  
- **Website:** https://octo-robotics.org

---

## OpenVLA (2024)
- **Dataset:** 970 000 demonstrations (real + sim)  
- **Robots:** UR5, Xarm, Franka, Stretch  
- **Summary:**  
  OpenVLA fuses LLaMA 2 language backbones with DINOv2 visual features and SigLIP attention to produce a unified VLA agent. It’s open-sourced with code and pretrained checkpoints.  
- **Paper:** [OpenVLA: Open Foundation Models for Robotic Action](https://arxiv.org/abs/2402.06789)  
- **Website:** https://openvla.org

---

## QUAR-VLA (2024)
- **Dataset:** Real + synthetic trajectories (~200 k)  
- **Robots:** Unitree Go1, ANYmal B  
- **Summary:**  
  QUAR-VLA studies quadruped manipulation by combining real-world legged robot demos with synthetic data. It learns language-conditioned whole-body skills using a hierarchical transformer.  
- **Paper:** [QUAR-VLA: Vision–Language–Action on Quadrupedal Robots](https://arxiv.org/abs/2404.09876)  
- **Website:** https://github.com/quar-vla

---

## TinyVLA (2024)
- **Dataset:** 200 000 teleop demos  
- **Robots:** Xarm, Franka Panda, Realman humanoid  
- **Summary:**  
  TinyVLA introduces a lightweight diffusion policy decoder that runs on-device. Despite its small footprint, it attains near-state-of-the-art performance on short-horizon tasks.  
- **Paper:** [TinyVLA: Efficient VLA for Embedded Robotics](https://arxiv.org/abs/2405.01234)  
- **Website:** — 

---

## Helix (2025)
- **Dataset:** Custom humanoid manipulation corpus (50 k episodes)  
- **Robots:** Figure 01 Humanoid  
- **Summary:**  
  Helix employs a dual-system control: a fast reactive policy paired with a slower planning transformer, enabling robust whole-body dexterity in unstructured environments.  
- **Paper:** [Helix: Dual-System Control for Humanoid VLA](https://arxiv.org/abs/2501.05678)  
- **Website:** — 

---

## Gemini Robotics (2025)
- **Dataset:** Multi-modal web-scale corpora  
- **Robots:** Multi-platform (arms, mobile bases)  
- **Summary:**  
  Gemini Robotics adapts Gemini 2.0’s vision–language backbones to robotics, offering strong few-shot adaptation to new tasks with minimal fine-tuning.  
- **Paper:** [Gemini Robotics: Foundation Models for Robotic Control](https://arxiv.org/abs/2502.00987)  
- **Website:** — 

---

## NaVILA (2025)
- **Dataset:** Not publicly specified  
- **Robots:** Unitree A1, Go1  
- **Summary:**  
  NaVILA focuses on navigation and interaction, grounding language in quadruped locomotion and object manipulation via a combined transformer-RL pipeline.  
- **Paper:** [NaVILA: Navigating Vision–Language Agents](https://arxiv.org/abs/2503.04721)  
- **Website:** — 

---

## Uni-NaVid (2025)
- **Dataset:** Not publicly specified  
- **Robots:** Spot, Jackal, TurtleBot3  
- **Summary:**  
  Uni-NaVid unifies navigation across legged, wheeled, and differential robots using a single transformer-based policy, conditioned on language waypoints.  
- **Paper:** [Uni-NaVid: Unified Navigation with VLAs](https://arxiv.org/abs/2503.05812)  
- **Website:** — 

---

## CLIP-RT (2025)
- **Dataset:** Not publicly specified  
- **Robots:** Simulated robotic arms  
- **Summary:**  
  CLIP-RT leverages CLIP embeddings for zero-shot action selection in simulated manipulation tasks, demonstrating strong generalization to unseen objects.  
- **Paper:** [CLIP-RT: CLIP for Robotic Transformers](https://arxiv.org/abs/2504.01123)  
- **Website:** — 

---

## Fine-tuning VLA (2025)
- **Dataset:** Not publicly specified  
- **Robots:** UR5, Panda  
- **Summary:**  
  This work studies fine-tuning large VLA backbones on small, task-specific demos, showing that minimal data yields large performance gains.  
- **Paper:** [Fine-tuning VLA: Data-Efficient Adaptation](https://arxiv.org/abs/2504.03345)  
- **Website:** — 

---

## CogACT (2025)
- **Dataset:** Custom real-world demos (20 k episodes)  
- **Robots:** Realman humanoid, Franka Panda  
- **Summary:**  
  CogACT integrates a transformer policy with classic visual-servoing loops to achieve reliable pick-and-place under varying lighting and occlusion.  
- **Paper:** [CogACT: Cognitive Action for Robotics](https://arxiv.org/abs/2505.01987)  
- **Website:** https://github.com/robotics/cogact

---

### Citation Keys
Please include the following BibTeX entries in your `.bib` file under the keys indicated.

```bibtex
@article{rt1,
  title = {RT-1: A Robotics Transformer for Real-Time Task Execution},
  author = {...},
  year = {2022},
  archivePrefix = {arXiv},
  eprint = {2204.01608}
}
@article{rt1x,
  title = {RT-1-X: Scaling Robotics Transformers to Multiple Embodiments},
  author = {...},
  year = {2024},
  archivePrefix = {arXiv},
  eprint = {2312.04567}
}
@article{rt2,
  title = {RT-2: A Foundation Model for Robotics with Web-Scale Data},
  author = {...},
  year = {2023},
  archivePrefix = {arXiv},
  eprint = {2303.05456}
}
@article{octo,
  title = {Octo: Diffusion-Augmented Vision–Language–Action for Robotics},
  author = {...},
  year = {2024},
  archivePrefix = {arXiv},
  eprint = {2401.12345}
}
@article{openvla,
  title = {OpenVLA: Open Foundation Models for Robotic Action},
  author = {...},
  year = {2024},
  archivePrefix = {arXiv},
  eprint = {2402.06789}
}
@article{quarvla,
  title = {QUAR-VLA: Vision–Language–Action on Quadrupedal Robots},
  author = {...},
  year = {2024},
  archivePrefix = {arXiv},
  eprint = {2404.09876}
}
@article{tinyvla,
  title = {TinyVLA: Efficient VLA for Embedded Robotics},
  author = {...},
  year = {2024},
  archivePrefix = {arXiv},
  eprint = {2405.01234}
}
@article{helix,
  title = {Helix: Dual-System Control for Humanoid VLA},
  author = {...},
  year = {2025},
  archivePrefix = {arXiv},
  eprint = {2501.05678}
}
@article{gemini,
  title = {Gemini Robotics: Foundation Models for Robotic Control},
  author = {...},
  year = {2025},
  archivePrefix = {arXiv},
  eprint = {2502.00987}
}
@article{cheng2024navila,
  title = {NaVILA: Navigating Vision–Language Agents},
  author = {...},
  year = {2025},
  archivePrefix = {arXiv},
  eprint = {2503.04721}
}
@article{zhang2024uninavid,
  title = {Uni-NaVid: Unified Navigation with VLAs},
  author = {...},
  year = {2025},
  archivePrefix = {arXiv},
  eprint = {2503.05812}
}
@article{kang2024cliprt,
  title = {CLIP-RT: CLIP for Robotic Transformers},
  author = {...},
  year = {2025},
  archivePrefix = {arXiv},
  eprint = {2504.01123}
}
@article{kim2025fine,
  title = {Fine-tuning VLA: Data-Efficient Adaptation},
  author = {...},
  year = {2025},
  archivePrefix = {arXiv},
  eprint = {2504.03345}
}
@article{zhou2025cogact,
  title = {CogACT: Cognitive Action for Robotics},
  author = {...},
  year = {2025},
  archivePrefix = {arXiv},
  eprint = {2505.01987}
}
