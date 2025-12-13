# VLA Models Evaluation & Visualization

Comprehensive analysis and visualization suite for Vision-Language-Action (VLA) models evaluation.

## Quick Start

### 1. Create Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Analysis
```bash
python final_plots.py
```

## Files

- **`final_plots.py`** - Main script generating all visualizations
- **`new_vla_models.csv`** - Dataset with 101 VLA models and evaluation metrics
- **`top75.csv`** - Subset with VLA-FEB component scores (CMAS, E_fusion, R2S, GI)

## Output

Plots are saved to:
- `plots/` - Publication-ready figures (PNG/SVG/PDF)
- Main plots include: forest plot, encoder analysis, domain analysis, fusion theory, VLA-FEB histogram

## Key Metrics

- **VLA-FEB Score**: Composite metric combining Cross-Modal Alignment (CMAS), Fusion Energy (E_fusion), Real-to-Sim Transfer (R2S), and Generalization Index (GI)
- **Adjusted Success**: Normalized task success rates (0-1 scale)
- **Generalization Index**: Multi-task capability measure
- **Difficulty Index**: Task complexity metric

## Requirements

- Python 3.10+
- See `requirements.txt` for package dependencies
