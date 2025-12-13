"""
Author: Dr. Muhayy Ud Din
Date: 2025-10-24
Input:  Table 1 (102 VLA models)
Output:
    analysis plots
"""

# ---------------------------------------------------------------------
# 0. Imports and enhanced style setup
# ---------------------------------------------------------------------
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from matplotlib import rcParams
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis

# Enhanced publication-quality styling
rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "axes.linewidth": 1.2,
    "grid.linewidth": 0.8,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.fancybox": True,
    "legend.shadow": True
})

# Enhanced seaborn style
sns.set_style("whitegrid", {
    'axes.grid': True, 
    'axes.edgecolor': '0.2',
    'grid.color': '0.9',
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Set color palette
sns.set_palette("husl")
os.makedirs("figs", exist_ok=True)
os.makedirs("Plot_script/plots", exist_ok=True)
os.makedirs("comment4", exist_ok=True)

# ---------------------------------------------------------------------
# 1. Hardcoded model list (from Table 1)
# ---------------------------------------------------------------------
models = [
    "CLIPort", "RT-1", "Gato", "VIMA", "PerAct", "SayCan", "RoboAgent",
    "RT-Trajectory", "ACT", "RT-2", "VoxPoser", "CLIP-RT", "Diffusion Policy",
    "Octo", "VLATest", "NaVILA", "RoboNurse-VLA", "Mobility VLA", "RevLA",
    "Uni-NaVid", "RDT-1B", "RoboMamba", "Chain-of-Affordance", "Edge VLA",
    "OpenVLA", "CogACT", "ShowUI-2B", "Pi-0", "HiRT", "A3VLM", "SVLR", "Bi-VLA",
    "QUAR-VLA", "3D-VLA", "RoboMM", "FAST", "OpenVLA-OFT", "CoVLA", "ORION",
    "UAV-VLA", "Combat VLA", "HybridVLA", "NORA", "SpatialVLA", "MoLe-VLA",
    "JARVIS-VLA", "UP-VLA", "Shake-VLA", "MORE", "DexGraspVLA", "DexVLA",
    "Humanoid-VLA", "ObjectVLA", "Gemini Robotics", "ECoT", "OTTER", "π-0.5",
    "OneTwoVLA", "Helix", "Gemini Robotics On-Device", "OE-VLA", "SmolVLA",
    "EF-VLA", "PD-VLA", "LeVERB", "TLA", "Interleave-VLA", "iRe-VLA",
    "TraceVLA", "OpenDrive VLA", "V-JEPA 2", "Knowledge Insulating VLA",
    "GR00T N1", "AgiBot World Colosseo", "Hi Robot", "EnerVerse", "FLaRe",
    "Beyond Sight", "GeoManip", "Universal Actions", "RoboHorizon", "SAM2Act",
    "LMM Planner Integration", "VLA-Cache", "Forethought VLA", "GRAPE",
    "HAMSTER", "TempoRep VLA", "ConRFT", "RoboBERT", "Diffusion Transformer Policy",
    "GEVRM", "SoFar", "ARM4R", "Magma", "An Atomic Skill Library", "VLAS",
    "ChatVLA", "RoboBrain", "SafeVLA", "CognitiveDrone", "Diffusion-VLA"
]

import re
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# 2. Load existing CSV data - USE ONLY ACTUAL DATA, NO FALLBACKS
# ---------------------------------------------------------------------
print("[✓] Loading existing vla_models.csv...")
df = pd.read_csv("new_vla_models.csv")
print(f"[✓] Loaded {len(df)} VLA model records")

# Verify all required columns exist
required_cols = ['FusionDepth', 'FusionType', 'DecoderFamily', 'Domain', 
                 'VisionParams', 'LLMParams', 'CTask', 'CMod', 'LogN', 'Adjusted_Success_0to1']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"ERROR: Missing required columns in CSV: {missing_cols}. Please ensure vla_models.csv has all required data.")

print("[✓] All required columns present in CSV")
print(f"[✓] Data ready with {len(df)} models and {len(df.columns)} features")

# Preview top few
print("\n[DATA PREVIEW]")
print(df.head(10).to_string())

# ---------------------------------------------------------------------
# 7. (Optional) Quick summary
# ---------------------------------------------------------------------
print("\nFusionDepth counts:")
print(df["FusionDepth"].value_counts())
print("\nFusionType counts:")
print(df["FusionType"].value_counts())
print("\nDecoderFamily counts:")
print(df["DecoderFamily"].value_counts())
print("\nDomain counts:")
print(df["Domain"].value_counts())

# ---------------------------------------------------------------------
# 3. Quantitative meta-analysis
# ---------------------------------------------------------------------
# Save categorical columns before encoding for later visualization
df_categorical = df[['Model', 'FusionDepth', 'FusionType', 'DecoderFamily', 'Domain']].copy()

# Categorize vision model size based on parameter count (before standardization)
def categorize_vision_size(params_log):
    """
    Categorize vision models into Small, Medium, Large based on log-scale parameters
    Level-1 (Small): ResNet18, MobileNet, ViT-Tiny → Basic perception
    Level-2 (Medium): ResNet50, ViT-Base, CLIP-RN50 → Solid semantic grounding
    Level-3 (Large): ViT-Large, CLIP-ViT-L, ViT-G/14, SAM → Human-level understanding
    """
    if params_log < 7.3:  # < ~20M parameters
        return 'Small'
    elif params_log < 8.3:  # 20M - 200M parameters
        return 'Medium'
    else:  # > 200M parameters
        return 'Large'

df['VisionModelSize'] = df['VisionParams'].apply(categorize_vision_size)

# Categorize language model size based on LLMParams value
def categorize_llm_size(llm_params):
    """
    Categorize language models into Small, Medium, Large based on log-scale parameters
    Level-1 (Small): CLIP-Text, T5-Base, BERT → Basic task understanding (< 7.8)
    Level-2 (Medium): LLaMA-2-7B, Qwen-7B, Gemma-7B → Stronger semantic interpretation (7.8-8.7)
    Level-3 (Large): GPT-3.5/4, Qwen-72B, LLaMA-70B → Advanced reasoning (> 8.7)
    """
    if llm_params < 7.8:  # < ~2B parameters
        return 'Small'
    elif llm_params <= 8.7:  # 2B - 70B parameters (most 7B models)
        return 'Medium'
    else:  # > 70B parameters
        return 'Large'

df['LLMSize'] = df['LLMParams'].apply(categorize_llm_size)

# Filter out 'mid' fusion depth - only keep early, late, hierarchical
df = df[df['FusionDepth'] != 'mid'].copy()

fusion_depth_map = {"early":1,"late":2,"hierarch":3}
df["D_f"] = df["FusionDepth"].map(fusion_depth_map).fillna(0)
df = pd.get_dummies(df, columns=["FusionType","DecoderFamily","Domain"], drop_first=True)

# Merge back the categorical columns for plotting
df = df.merge(df_categorical, on='Model', suffixes=('', '_cat'))

# Save original values before standardization for entropy calculations
df['CMod_orig'] = df['CMod'].copy()
df['CTask_orig'] = df['CTask'].copy()
df['VisionParams_orig'] = df['VisionParams'].copy()
df['LLMParams_orig'] = df['LLMParams'].copy()
df['Difficulty_Index_orig'] = df['Difficulty_Index'].copy()

scaler = StandardScaler()
num_cols = ["D_f","VisionParams","LLMParams","CTask","CMod","LogN"]
df[num_cols] = scaler.fit_transform(df[num_cols])

# Create binary indicator for hierarchical fusion
df['Fusion_hier'] = (df['FusionDepth_cat'] == 'hierarch').astype(int)

# Regression model matching Equation A.1
# Y = β0 + β1*D_f + β2*S_v + β3*S_l + β4*C_task + β5*C_mod 
#     + β6*Decoder_diff + β7*Decoder_flow + β8*Fusion_hier + u_bench + ε
formula = "Adjusted_Success_0to1 ~ D_f + VisionParams + LLMParams + CTask + CMod + DecoderFamily_diffusion + DecoderFamily_flow + Fusion_hier"
model = smf.ols(formula, data=df).fit()
coef = model.params.drop("Intercept")
conf_int = model.conf_int().drop("Intercept")
stderr = model.bse.drop("Intercept")
print(model.summary())

# Print available columns for debugging
print(f"Available columns: {df.columns.tolist()}")

# ---------------------------------------------------------------------
# 4. Enhanced Visualization (IEEE/LaTeX style with improved quality)
# ---------------------------------------------------------------------

# 1. Enhanced Forest Plot (showing absolute values with grouped categories)
# Define label mapping matching Equation A.1
label_mapping = {
    'D_f': r'$D_f$',
    'VisionParams': r'$S_v$',
    'LLMParams': r'$S_\ell$',
    'CTask': r'$C_{\mathrm{task}}$',
    'CMod': r'$C_{\mathrm{mod}}$',
    'DecoderFamily_diffusion[T.True]': r'$\mathbb{I}_{\mathrm{diffusion}}$',
    'DecoderFamily_flow[T.True]': r'$\mathbb{I}_{\mathrm{flow}}$',
    'Fusion_hier': r'$\mathbb{I}_{\mathrm{hierarchical}}$'
}

# Select all coefficients from Equation A.1
selected_vars = ['D_f', 'VisionParams', 'LLMParams', 'CTask', 'CMod',
                 'DecoderFamily_diffusion[T.True]', 'DecoderFamily_flow[T.True]', 
                 'Fusion_hier']

# Filter coefficients
coef_filtered = coef[[var for var in selected_vars if var in coef.index]]

# Create groups for visualization matching equation structure
groups = {
    'Architecture Design': ['D_f', 'Fusion_hier'],
    'Model Scale': ['VisionParams', 'LLMParams'],
    'Task Complexity': ['CTask', 'CMod'],
    'Decoder Policy': ['DecoderFamily_flow[T.True]', 'DecoderFamily_diffusion[T.True]']
}

# Create ordered list based on groups (order matching equation)
ordered_vars = []
for group_name in ['Architecture Design', 'Model Scale', 'Task Complexity', 'Decoder Policy']:
    ordered_vars.extend([var for var in groups[group_name] if var in coef_filtered.index])

coef_ordered = coef_filtered[ordered_vars]

plt.figure(figsize=(10, 5))
ax = plt.gca()

# Use actual coefficient values (not absolute)
y_pos = np.arange(len(coef_ordered))

# Add group separators and background colors first
group_positions = []
current_pos = 0
for group_name in ['Architecture Design', 'Model Scale', 'Task Complexity', 'Decoder Policy']:
    group_size = len([var for var in groups[group_name] if var in coef_filtered.index])
    if group_size > 0:
        group_positions.append((current_pos, group_size, group_name))
        current_pos += group_size

# Distinct colors for each group (professional palette - darker versions)
group_colors = {
    'Architecture Design': '#5DACA8',      # Darker Teal/Cyan
    'Model Scale': '#E85D48',              # Darker Coral/Salmon
    'Task Complexity': '#8BC34A',          # Darker Yellow-Green
    'Decoder Policy': '#9B8AC4'            # Darker Lavender/Purple
}

# Background colors for each group (lighter versions)
group_bg_colors = {
    'Architecture Design': '#E5F5F4',      # Very light teal
    'Model Scale': '#FEE9E7',              # Very light coral
    'Task Complexity': '#F0F4E5',          # Very light yellow-green
    'Decoder Policy': '#F0EDF7'            # Very light lavender
}

# Assign colors based on which group each coefficient belongs to
bar_colors = []
for idx in coef_ordered.index:
    clean_idx = idx.replace('[T.True]', '')
    for group_name, vars_list in groups.items():
        if idx in vars_list:
            bar_colors.append(group_colors[group_name])
            break

# Add background colors for each group
for i, (start_pos, size, name) in enumerate(group_positions):
    ax.axhspan(start_pos - 0.5, start_pos + size - 0.5, 
               facecolor=group_bg_colors[name], alpha=0.4, zorder=0)

# Add vertical line at x=0
plt.axvline(x=0, color='gray', linewidth=1.5, linestyle='--', alpha=0.6, zorder=1)

# Get confidence intervals for the ordered coefficients
conf_int_ordered = conf_int.loc[coef_ordered.index]
lower_bounds = conf_int_ordered[0].values
upper_bounds = conf_int_ordered[1].values

# Plot error bars (confidence intervals)
for i, (coef_val, lower, upper) in enumerate(zip(coef_ordered.values, lower_bounds, upper_bounds)):
    # Get color for this coefficient's group
    color = bar_colors[i]
    
    # Plot horizontal line for confidence interval
    ax.plot([lower, upper], [i, i], color=color, linewidth=3, alpha=0.7, zorder=2)
    
    # Plot point estimate as a larger marker
    ax.plot(coef_val, i, 'o', color=color, markersize=10, markeredgecolor='black', 
            markeredgewidth=1.5, alpha=0.95, zorder=3)

# Add value labels at the point estimates
for i, val in enumerate(coef_ordered.values):
    # Place text to the right for all values for consistency
    plt.text(val + 0.015, i, f'{val:.3f}', 
             ha='left', va='center', fontweight='bold', fontsize=9, color='black')

# Create improved labels without sign indicators (sign shown by bar direction)
labels_clean = []
for idx in coef_ordered.index:
    # First try the full index name, then try cleaned version
    if idx in label_mapping:
        label = label_mapping[idx]
    else:
        # Clean up the index name and try again
        clean_idx = idx.replace('[T.True]', '')
        label = label_mapping.get(clean_idx, clean_idx.replace('_', ' ').title())
    labels_clean.append(label)

plt.yticks(y_pos, labels_clean, fontsize=11)

# Set y-axis limits to remove extra space
plt.ylim(-0.5, len(coef_ordered) - 0.5)

plt.xlabel("Standardized Coefficients (β)", fontsize=11, fontweight='bold')

# Add legend with group colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=group_colors['Architecture Design'], label='Architecture Design', alpha=0.9, edgecolor='black'),
    Patch(facecolor=group_colors['Model Scale'], label='Model Scale', alpha=0.9, edgecolor='black'),
    Patch(facecolor=group_colors['Task Complexity'], label='Task Complexity', alpha=0.9, edgecolor='black'),
    Patch(facecolor=group_colors['Decoder Policy'], label='Decoder Policy', alpha=0.9, edgecolor='black')
]
plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("Plot_script/plots/forest_plot.svg", dpi=300, bbox_inches='tight')
plt.savefig("Plot_script/plots/forest_plot.png", dpi=300, bbox_inches='tight')
#plt.show()

# 2b. New 3-Panel Scale Analysis (1 row x 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Normalize raw Success to [0, 1] range based on all VLAs in the dataset
min_success = df['Success'].min()
max_success = df['Success'].max()
df['Normalized_Success'] = (df['Success'] - min_success) / (max_success - min_success)

# Plot 1: Vision Model Scale Impact
vision_size_order = ['Small', 'Medium', 'Large']
sns.boxplot(x='VisionModelSize', y='Normalized_Success', data=df, ax=axes[0], 
           order=vision_size_order, palette='pastel', hue='VisionModelSize', legend=False, width=0.5)
axes[0].set_title("Vision Model Scale Impact", fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel("Vision Model Size", fontsize=15, labelpad=12)
axes[0].set_ylabel("Normalized Success", fontsize=15, labelpad=12)
axes[0].tick_params(axis='both', labelsize=14, rotation=0, pad=8)

# Plot 2: Language Model Scale Impact
llm_size_order = ['Small', 'Medium', 'Large']
sns.boxplot(x='LLMSize', y='Normalized_Success', data=df, ax=axes[1], 
           order=llm_size_order, palette='Set3', hue='LLMSize', legend=False, width=0.5)
axes[1].set_title("Language Model Scale Impact", fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel("Language Model Size", fontsize=15, labelpad=12)
axes[1].set_ylabel("Normalized Success", fontsize=15, labelpad=12)
axes[1].tick_params(axis='both', labelsize=14, rotation=0, pad=8)

# Plot 3: Fusion Depth Impact (early, late, hierarchical only)
fusion_order = ['early', 'late', 'hierarch']
sns.boxplot(x='FusionDepth', y='Normalized_Success', data=df, ax=axes[2], 
           order=fusion_order, palette='Pastel2', hue='FusionDepth', legend=False, width=0.5)
axes[2].set_title("Fusion Depth Impact", fontsize=16, fontweight='bold', pad=15)
axes[2].set_xlabel("Fusion Depth", fontsize=15, labelpad=12)
axes[2].set_ylabel("Normalized Success", fontsize=15, labelpad=12)
axes[2].tick_params(axis='both', labelsize=14, rotation=0, pad=8)

# Add horizontal gridlines to all subplots
for ax in axes:
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.8)
    # Add multiple horizontal reference lines
    for y_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.axhline(y=y_val, color='gray', linestyle='--', linewidth=1.0, alpha=0.4, zorder=0)

plt.tight_layout()
plt.savefig("Plot_script/plots/scale_analysis_4panel.svg", dpi=300, bbox_inches='tight')
plt.savefig("Plot_script/plots/scale_analysis_4panel.png", dpi=300, bbox_inches='tight')
#plt.show()

# 2c. New 3-Panel Scale Analysis using Adjusted_Success_0to1 (1 row x 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: Vision Model Scale Impact
vision_size_order = ['Small', 'Medium', 'Large']
sns.boxplot(x='VisionModelSize', y='Adjusted_Success_0to1', data=df, ax=axes[0], 
           order=vision_size_order, palette='pastel', hue='VisionModelSize', legend=False, width=0.5)
axes[0].set_title("Vision Model Scale Impact", fontsize=16, fontweight='bold', pad=15)
axes[0].set_xlabel("Vision Model Size", fontsize=15, labelpad=12)
axes[0].set_ylabel("Adjusted Success (0-1)", fontsize=15, labelpad=12)
axes[0].tick_params(axis='both', labelsize=14, rotation=0, pad=8)

# Plot 2: Language Model Scale Impact
llm_size_order = ['Small', 'Medium', 'Large']
sns.boxplot(x='LLMSize', y='Adjusted_Success_0to1', data=df, ax=axes[1], 
           order=llm_size_order, palette='Set3', hue='LLMSize', legend=False, width=0.5)
axes[1].set_title("Language Model Scale Impact", fontsize=16, fontweight='bold', pad=15)
axes[1].set_xlabel("Language Model Size", fontsize=15, labelpad=12)
axes[1].set_ylabel("Adjusted Success (0-1)", fontsize=15, labelpad=12)
axes[1].tick_params(axis='both', labelsize=14, rotation=0, pad=8)

# Plot 3: Fusion Depth Impact (early, late, hierarchical only)
fusion_order = ['early', 'late', 'hierarch']
sns.boxplot(x='FusionDepth', y='Adjusted_Success_0to1', data=df, ax=axes[2], 
           order=fusion_order, palette='Pastel2', hue='FusionDepth', legend=False, width=0.5)
axes[2].set_title("Fusion Depth Impact", fontsize=16, fontweight='bold', pad=15)
axes[2].set_xlabel("Fusion Depth", fontsize=15, labelpad=12)
axes[2].set_ylabel("Adjusted Success (0-1)", fontsize=15, labelpad=12)
axes[2].tick_params(axis='both', labelsize=14, rotation=0, pad=8)

# Add horizontal gridlines to all subplots
for ax in axes:
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.8)
    # Add multiple horizontal reference lines
    for y_val in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.axhline(y=y_val, color='gray', linestyle='--', linewidth=1.0, alpha=0.4, zorder=0)

plt.tight_layout()
plt.savefig("Plot_script/plots/scale_analysis_adjusted_4panel.svg", dpi=300, bbox_inches='tight')
plt.savefig("Plot_script/plots/scale_analysis_adjusted_4panel.png", dpi=300, bbox_inches='tight')
print("[✓] Scale analysis with Adjusted_Success_0to1 saved: Plot_script/plots/scale_analysis_adjusted_4panel.png/svg")
#plt.show()

# 3. Create diffusion indicator for later use
# Create diffusion vs other classification
diffusion_cols = [c for c in df.columns if "diffusion" in c.lower() and c.startswith("DecoderFamily")]
if diffusion_cols:
    diffusion_col = diffusion_cols[0]
    df['IsDiffusion'] = df[diffusion_col].map({True: 'Diffusion', False: 'Other'})
    
    # Print statistical summary
    diffusion_success = df[df['IsDiffusion'] == 'Diffusion']['Adjusted_Success_0to1']
    other_success = df[df['IsDiffusion'] == 'Other']['Adjusted_Success_0to1']
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(diffusion_success, other_success)
    
    print("\n" + "="*60)
    print("DIFFUSION POLICY vs OTHER DECODERS - STATISTICAL SUMMARY")
    print("="*60)
    print(f"\nDiffusion Models: {len(diffusion_success)} ({len(diffusion_success)/len(df)*100:.1f}%)")
    print(f"Other Models: {len(other_success)} ({len(other_success)/len(df)*100:.1f}%)")
    
    print(f"\nSuccess Rate Statistics:")
    print(f"Diffusion - Mean: {diffusion_success.mean():.4f} ± {diffusion_success.std():.4f}")
    print(f"Other - Mean: {other_success.mean():.4f} ± {other_success.std():.4f}")
    print(f"Difference: {diffusion_success.mean() - other_success.mean():.4f}")
    print(f"T-test p-value: {p_value:.6f}")
    
    if p_value < 0.05:
        print("*** SIGNIFICANT DIFFERENCE (p < 0.05) ***")
    else:
        print("No significant difference (p >= 0.05)")
        

# 4. Vision and Language Encoder Categorization
# Extract vision encoder families from Vision_Encoder column
def categorize_vision_encoder(encoder_text):
    """Categorize vision encoders into families - separating pretrained foundation models from custom architectures"""
    encoder_text = str(encoder_text).lower()
    
    # Pretrained Vision-Language Foundation Models (check these first, before generic patterns)
    if 'pali' in encoder_text or 'palm-e' in encoder_text:  # PaLI-X, PaLM-E, PaliGemma
        return 'PaLI/PaLM'
    elif 'gemini' in encoder_text:  # Gemini 1.5, 2.0, SDK
        return 'Gemini'
    elif 'qwen' in encoder_text:  # Qwen2-VL, Qwen-VL, etc.
        return 'Qwen-VL'
    elif 'owl-vit' in encoder_text:  # OWL-ViT
        return 'OWL-ViT'
    elif 'florence' in encoder_text:  # Florence-2
        return 'Florence'
    elif 'sam' in encoder_text and 'sam' not in 'same':  # SAM, SAM2 (avoid "same as")
        return 'SAM'
    elif 'eva' in encoder_text:  # EVA-CLIP, EVA-02
        return 'EVA'
    elif 'molmo' in encoder_text:  # Molmo-7B
        return 'Molmo'
    elif 'llava' in encoder_text or 'vila' in encoder_text:  # LLaVA variants, VILA
        return 'LLaVA/VILA'
    elif 'phi' in encoder_text and 'vision' in encoder_text:  # Phi-3-Vision
        return 'Phi-Vision'
    elif 'blip' in encoder_text:  # BLIP-2, InstructBLIP
        return 'BLIP'
    
    # Established Vision Foundation Models (not VLMs)
    elif 'clip' in encoder_text:
        return 'CLIP'
    elif 'dino' in encoder_text or 'dinov2' in encoder_text:
        return 'DINOv2'
    elif 'siglip' in encoder_text:
        return 'SigLIP'
    
    # CNN Architectures
    elif 'efficientnet' in encoder_text:
        return 'EfficientNet'
    elif 'resnet' in encoder_text:
        return 'ResNet'
    elif 'convnext' in encoder_text:
        return 'ConvNeXt'
    elif 'cnn' in encoder_text or 'conv' in encoder_text:
        return 'CNN-based'
    
    # Custom/Generic Transformers (no pretrained foundation model)
    elif 'vit' in encoder_text or 'transformer' in encoder_text:
        return 'Custom ViT'
    
    else:
        return 'Other'

df['VisionEncoderFamily'] = df['Vision_Encoder'].apply(categorize_vision_encoder)

# For humanoid domain: convert "Other" to "Custom Encoders" to make it more informative
df.loc[(df['Domain'] == 'humanoid') & (df['VisionEncoderFamily'] == 'Other'), 'VisionEncoderFamily'] = 'Custom Encoders'

# Extract language model families
def categorize_language_model(model_text):
    """Categorize language models into families - separating pretrained foundation models from custom architectures"""
    model_text = str(model_text).lower()
    
    # Handle missing values
    if 'none' in model_text or 'nan' in model_text or model_text == 'nan':
        return 'None/Other'
    
    # Pretrained Language Model Foundation Families (check specific ones first)
    if 'llama' in model_text:  # LLaMA, LLaMA-2, LLaMA-3, Llama variants
        return 'LLaMA'
    elif 'qwen' in model_text:  # Qwen, Qwen2, Qwen2-VL, Qwen2.5
        return 'Qwen'
    elif 'gemma' in model_text or 'gemini' in model_text:  # Gemma, Gemini, PaliGemma
        return 'Gemma'
    elif 'phi' in model_text:  # Phi-1.5, Phi-3, Phi-3mini
        return 'Phi'
    elif 'gpt' in model_text:  # GPT-3, GPT-4, GPT-4o
        return 'GPT'
    elif 'palm' in model_text and 'pali' not in model_text:  # PaLM (but not PaLI-X which is vision-language)
        return 'PaLM'
    elif 'vicuna' in model_text:  # Vicuna variants (fine-tuned LLaMA)
        return 'Vicuna'
    elif 't5' in model_text or 'flan' in model_text:  # T5, T5-XXL, Flan-T5
        return 'T5/Flan'
    elif 'mistral' in model_text:  # Mistral
        return 'Mistral'
    elif 'mpt' in model_text:  # MPT
        return 'MPT'
    elif 'pythia' in model_text:  # Pythia
        return 'Pythia'
    elif 'bert' in model_text:  # BERT variants
        return 'BERT'
    elif 'fuyu' in model_text:  # Fuyu
        return 'Fuyu'
    elif 'starling' in model_text:  # Starling-LM
        return 'Starling'
    elif 'smol' in model_text:  # SmolLM, SmolVLM
        return 'SmolLM/VLM'
    
    # Vision-Language Model Text Encoders
    elif 'clip' in model_text and 'text' in model_text:  # CLIP text encoder
        return 'CLIP Text'
    elif 'siglip' in model_text and 'text' in model_text:  # SigLIP text encoder
        return 'SigLIP Text'
    elif 'llava' in model_text or 'vila' in model_text:  # LLaVA, VILA
        return 'LLaVA/VILA'
    elif 'blip' in model_text:  # BLIP-2
        return 'BLIP'
    elif 'pali' in model_text:  # PaLI-X, PaliGemma text components
        return 'PaLI'
    
    # Custom/Generic Transformers
    elif 'transformer' in model_text:
        return 'Cust. Transformer'
    
    else:
        return 'Other'

df['LanguageModelFamily'] = df['Language_Encoder'].apply(categorize_language_model)

# Print statistical summaries
print("\n" + "="*70)
print("VISION ENCODER PERFORMANCE ANALYSIS")
print("="*70)
vision_success = df.groupby('VisionEncoderFamily')['Adjusted_Success_0to1'].agg(['mean', 'std', 'count']).reset_index()
vision_success = vision_success.sort_values('mean', ascending=False)
print(vision_success.round(4))

print("\n" + "="*70)
print("LANGUAGE MODEL PERFORMANCE ANALYSIS")
print("="*70)
lang_success = df.groupby('LanguageModelFamily')['Adjusted_Success_0to1'].agg(['mean', 'std', 'count']).reset_index()
lang_success = lang_success.sort_values('mean', ascending=False)
print(lang_success.round(4))

# Statistical tests
from scipy.stats import f_oneway

print("\n" + "="*70)
print("STATISTICAL SIGNIFICANCE TESTS")
print("="*70)

# ANOVA for vision encoders
vision_groups = [group['Adjusted_Success_0to1'].values for name, group in df.groupby('VisionEncoderFamily')]
f_stat_vision, p_val_vision = f_oneway(*vision_groups)
print(f"Vision Encoder ANOVA: F={f_stat_vision:.4f}, p={p_val_vision:.6f}")

# ANOVA for language models
lang_groups = [group['Adjusted_Success_0to1'].values for name, group in df.groupby('LanguageModelFamily')]
f_stat_lang, p_val_lang = f_oneway(*lang_groups)
print(f"Language Model ANOVA: F={f_stat_lang:.4f}, p={p_val_lang:.6f}")

if p_val_vision < 0.05:
    print("*** SIGNIFICANT DIFFERENCES between Vision Encoders (p < 0.05) ***")
else:
    print("No significant differences between Vision Encoders (p >= 0.05)")
    
if p_val_lang < 0.05:
    print("*** SIGNIFICANT DIFFERENCES between Language Models (p < 0.05) ***")
else:
    print("No significant differences between Language Models (p >= 0.05)")

# 5. Enhanced Factor analysis (latent structure) with visualization
# ---------------------------------------------------------------------
fa = FactorAnalysis(n_components=3, random_state=0)  # Increased to 3 factors
latent = fa.fit_transform(df[num_cols])

# Create label mapping for factor analysis features
feature_label_mapping = {
    'D_f': 'Fusion Depth',
    'VisionParams': 'Vision Model Size',
    'LLMParams': 'Language Model Size',
    'CTask': 'Task Difficulty',
    'CMod': 'Sensor Modalities',
    'LogN': 'Dataset Size'
}

# Apply label mapping to feature names
improved_labels = [feature_label_mapping.get(col, col) for col in num_cols]

loadings = pd.DataFrame(
    fa.components_.T,
    index=improved_labels,
    columns=["Factor1_Architecture", "Factor2_Scale", "Factor3_Performance"]
)

# Enhanced Factor Loading Visualization
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(loadings.T, annot=True, cmap='RdBu_r', center=0, 
            square=True, cbar_kws={'label': 'Factor Loading', 'shrink': 0.6}, ax=ax)
plt.xlabel("Model Features", fontsize=11, fontweight='bold')
plt.ylabel("Latent Factors", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("Plot_script/plots/factor_analysis.svg", dpi=300, bbox_inches='tight')
plt.savefig("Plot_script/plots/factor_analysis.png", dpi=300, bbox_inches='tight')
#plt.show()

# Save enhanced loadings
loadings.round(3).to_csv("Plot_script/plots/factor_loadings.csv")

print(f"""
[✓] Enhanced Visualizations Generated:
    • Plot_script/plots/forest_plot.svg & .png - Enhanced coefficient forest plot
    • Plot_script/plots/partial_dependence.svg & .png - Multi-panel relationship analysis  
    • Plot_script/plots/factor_analysis.svg & .png - Factor loading heatmap
    • Plot_script/plots/factor_loadings.csv - Enhanced factor loadings
    
[✓] All plots now feature:
    • High resolution (300 DPI)
    • Enhanced readability with larger fonts
    • Professional color schemes
    • Improved layout and spacing
    • Both SVG and PNG formats
    • Statistical annotations and trend lines
""")

# ---------------------------------------------------------------------
# 8. Combined 4-Panel Vision & Language Encoder Analysis
# ---------------------------------------------------------------------
print("\n[✓] Generating combined 4-panel encoder analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
# fig.suptitle("Vision and Language Encoder Performance Analysis", fontsize=20, fontweight='bold', y=0.98)

# Panel 1: Success Rate by Vision Encoder Family (top-left)
vision_success = df.groupby('VisionEncoderFamily')['Adjusted_Success_0to1'].agg(['mean', 'std', 'count']).reset_index()
vision_success = vision_success[vision_success['VisionEncoderFamily'] != 'Other']  # Remove "Other"
vision_success = vision_success[vision_success['VisionEncoderFamily'] != 'CNN-based']  # Remove "CNN-based"
vision_success = vision_success[vision_success['count'] >= 3]  # Only include families with n >= 3
vision_success = vision_success.sort_values('mean', ascending=False)

bars1 = axes[0,0].bar(vision_success['VisionEncoderFamily'], vision_success['mean'], 
                    yerr=vision_success['std'], capsize=5,
                    color=sns.color_palette("viridis", len(vision_success)),
                    alpha=0.8, edgecolor='black')
axes[0,0].set_title("(a) Success Rate by Vision Encoder Family", fontsize=16, fontweight='bold')
axes[0,0].set_ylabel("Normalized Success", fontsize=14)
axes[0,0].tick_params(axis='x', rotation=45, labelsize=12)
axes[0,0].tick_params(axis='y', labelsize=12)
axes[0,0].grid(True, alpha=0.3)

# Add value labels
for i, bar in enumerate(bars1):
    height = bar.get_height()
    axes[0,0].text(bar.get_x() + 0.05, 0.02,
                  f'{height:.3f}', ha='left', va='bottom', fontweight='bold', fontsize=11)

# Panel 2: Generalization Index by Vision Encoder (top-right)
if 'Generalization_Index_0to1' in df.columns:
    vision_gen = df.groupby('VisionEncoderFamily')['Generalization_Index_0to1'].agg(['mean', 'std', 'count']).reset_index()
    vision_gen = vision_gen[vision_gen['VisionEncoderFamily'] != 'Other']  # Remove "Other"
    vision_gen = vision_gen[vision_gen['VisionEncoderFamily'] != 'CNN-based']  # Remove "CNN-based"
    vision_gen = vision_gen[vision_gen['count'] >= 3]  # Only include families with n >= 3
    vision_gen = vision_gen.sort_values('mean', ascending=False)
    
    bars2 = axes[0,1].bar(vision_gen['VisionEncoderFamily'], vision_gen['mean'], 
                        yerr=vision_gen['std'], capsize=5,
                        color=sns.color_palette("cividis", len(vision_gen)),
                        alpha=0.8, edgecolor='black')
    axes[0,1].set_title("(b) Generalization Index by Vision Encoder", fontsize=16, fontweight='bold')
    axes[0,1].set_ylabel("Generalization Index", fontsize=14)
    axes[0,1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[0,1].tick_params(axis='y', labelsize=12)
    axes[0,1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + 0.05, 0.02,
                      f'{height:.3f}', ha='left', va='bottom', fontweight='bold', fontsize=11)
else:
    axes[0,1].text(0.5, 0.5, 'Generalization_Index_0to1\ncolumn not found', 
                   ha='center', va='center', fontsize=14)
    axes[0,1].set_title("(b) Generalization Index by Vision Encoder", fontsize=14, fontweight='bold')

# Panel 3: Success Rate by Language Model Family (bottom-left)
lang_success = df.groupby('LanguageModelFamily')['Adjusted_Success_0to1'].agg(['mean', 'std', 'count']).reset_index()
lang_success = lang_success[~lang_success['LanguageModelFamily'].isin(['Other', 'None/Other'])]  # Remove "Other" and "None/Other"
lang_success = lang_success[lang_success['count'] >= 3]  # Only include families with n >= 3
lang_success = lang_success.sort_values('mean', ascending=False)

bars3 = axes[1,0].bar(lang_success['LanguageModelFamily'], lang_success['mean'], 
                    yerr=lang_success['std'], capsize=5,
                    color=sns.color_palette("plasma", len(lang_success)),
                    alpha=0.8, edgecolor='black')
axes[1,0].set_title("(c) Success Rate by Language Model Family", fontsize=16, fontweight='bold')
axes[1,0].set_ylabel("Normalized Success", fontsize=14)
axes[1,0].tick_params(axis='x', rotation=45, labelsize=12)
axes[1,0].tick_params(axis='y', labelsize=12)
axes[1,0].grid(True, alpha=0.3)

# Add value labels
for i, bar in enumerate(bars3):
    height = bar.get_height()
    axes[1,0].text(bar.get_x() + 0.05, 0.02,
                  f'{height:.3f}', ha='left', va='bottom', fontweight='bold', fontsize=11)

# Panel 4: Generalization Index by Language Model (bottom-right)
if 'Generalization_Index_0to1' in df.columns:
    lang_gen = df.groupby('LanguageModelFamily')['Generalization_Index_0to1'].agg(['mean', 'std', 'count']).reset_index()
    lang_gen = lang_gen[~lang_gen['LanguageModelFamily'].isin(['Other', 'None/Other'])]  # Remove "Other" and "None/Other"
    lang_gen = lang_gen[lang_gen['count'] >= 3]  # Only include families with n >= 3
    lang_gen = lang_gen.sort_values('mean', ascending=False)
    
    bars4 = axes[1,1].bar(lang_gen['LanguageModelFamily'], lang_gen['mean'], 
                        yerr=lang_gen['std'], capsize=5,
                        color=sns.color_palette("magma", len(lang_gen)),
                        alpha=0.8, edgecolor='black')
    axes[1,1].set_title("(d) Generalization Index by Language Model", fontsize=16, fontweight='bold')
    axes[1,1].set_ylabel("Generalization Index", fontsize=14)
    axes[1,1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1,1].tick_params(axis='y', labelsize=12)
    axes[1,1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + 0.05, 0.02,
                      f'{height:.3f}', ha='left', va='bottom', fontweight='bold', fontsize=11)
else:
    axes[1,1].text(0.5, 0.5, 'Generalization_Index_0to1\ncolumn not found', 
                   ha='center', va='center', fontsize=14)
    axes[1,1].set_title("(d) Generalization Index by Language Model", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("Plot_script/plots/encoder_analysis_4panel.svg", dpi=300, bbox_inches='tight')
plt.savefig("Plot_script/plots/encoder_analysis_4panel.png", dpi=300, bbox_inches='tight')
print("[✓] Combined 4-panel encoder analysis saved: Plot_script/plots/encoder_analysis_4panel.png/svg")
#plt.show()

# ---------------------------------------------------------------------
# 9. Decoder Analysis: 2-Panel (Success & Generalization Index)
# ---------------------------------------------------------------------
print("\n[✓] Generating 2-panel decoder analysis...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
# fig.suptitle("Action Decoder Performance Analysis", fontsize=20, fontweight='bold', y=1.02)

# Panel 1: Normalized Success by Decoder Family (left)
decoder_success = df.groupby('DecoderFamily')['Adjusted_Success_0to1'].agg(['mean', 'std', 'count']).reset_index()
decoder_success = decoder_success[decoder_success['DecoderFamily'] != 'flow']  # Remove "flow" (only 2 models)
decoder_success = decoder_success.sort_values('mean', ascending=False)

bars1 = axes[0].bar(decoder_success['DecoderFamily'], decoder_success['mean'], 
                    yerr=decoder_success['std'], capsize=5,
                    color=sns.color_palette("rocket", len(decoder_success)),
                    alpha=0.8, edgecolor='black')
axes[0].set_title("(a) Success Rate by Decoder Family", fontsize=16, fontweight='bold')
axes[0].set_ylabel("Normalized Success", fontsize=14)
axes[0].set_xlabel("Decoder Family", fontsize=14)
axes[0].tick_params(axis='x', rotation=45, labelsize=12)
axes[0].tick_params(axis='y', labelsize=12)
axes[0].grid(True, alpha=0.3)

# Add value labels
for i, bar in enumerate(bars1):
    height = bar.get_height()
    axes[0].text(bar.get_x() + 0.05, 0.02,
                f'{height:.3f}', ha='left', va='bottom', fontweight='bold', fontsize=11)

# Panel 2: Generalization Index by Decoder Family (right)
if 'Generalization_Index_0to1' in df.columns:
    decoder_gen = df.groupby('DecoderFamily')['Generalization_Index_0to1'].agg(['mean', 'std', 'count']).reset_index()
    decoder_gen = decoder_gen[decoder_gen['DecoderFamily'] != 'flow']  # Remove "flow" (only 2 models)
    decoder_gen = decoder_gen.sort_values('mean', ascending=False)
    
    bars2 = axes[1].bar(decoder_gen['DecoderFamily'], decoder_gen['mean'], 
                        yerr=decoder_gen['std'], capsize=5,
                        color=sns.color_palette("mako", len(decoder_gen)),
                        alpha=0.8, edgecolor='black')
    axes[1].set_title("(b) Generalization Index by Decoder Family", fontsize=16, fontweight='bold')
    axes[1].set_ylabel("Generalization Index", fontsize=14)
    axes[1].set_xlabel("Decoder Family", fontsize=14)
    axes[1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[1].text(bar.get_x() + 0.05, 0.02,
                    f'{height:.3f}', ha='left', va='bottom', fontweight='bold', fontsize=11)
else:
    axes[1].text(0.5, 0.5, 'Generalization_Index_0to1\ncolumn not found', 
                 ha='center', va='center', fontsize=14)
    axes[1].set_title("(b) Generalization Index by Decoder Family", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig("Plot_script/plots/decoder_analysis_2panel.svg", dpi=300, bbox_inches='tight')
plt.savefig("Plot_script/plots/decoder_analysis_2panel.png", dpi=300, bbox_inches='tight')
print("[✓] 2-panel decoder analysis saved: Plot_script/plots/decoder_analysis_2panel.png/svg")
#plt.show()

# ---------------------------------------------------------------------
# 11. Domain-wise Component Analysis: 4-Panel Figure (First row of 8-panel plot)
# ---------------------------------------------------------------------
print("\n[✓] Generating domain-wise component analysis...")

# Filter domains with sufficient data (n >= 3)
domain_counts = df['Domain'].value_counts()
domains_to_include = domain_counts[domain_counts >= 3].index.tolist()
df_domain = df[df['Domain'].isin(domains_to_include)].copy()

# Remove 'flow' decoder (only 2 models)
df_domain = df_domain[df_domain['DecoderFamily'] != 'flow'].copy()

# Create IsDiffusion column
df_domain['IsDiffusion'] = df_domain['DecoderFamily'].apply(lambda x: 'Diffusion' if x == 'diffusion' else 'Other')

# Define very light background colors for each domain
domain_colors = {
    'manipulation': '#E8F4F8',  # Very light blue
    'navigation': '#FFF4E6',    # Very light orange
    'humanoid': '#E8F5E9',      # Very light green
    'gui': '#FCE4EC'            # Very light pink
}

fig, axes = plt.subplots(1, 4, figsize=(28, 7))
# fig.suptitle("Domain-wise Component Performance Analysis", fontsize=24, fontweight='bold', y=1.02)

# Panel 1a: All Decoders - Success Rate
decoder_domain_success = df_domain.groupby(['Domain', 'DecoderFamily'])['Adjusted_Success_0to1'].mean().unstack(fill_value=0)
decoder_domain_success.plot(kind='bar', ax=axes[0], width=0.8, 
                            color=sns.color_palette("Set2", len(decoder_domain_success.columns)), legend=False)
axes[0].set_title("(a) Decoder Success Rate", fontsize=18, fontweight='bold')
axes[0].set_ylabel("Normalized Success", fontsize=16)
axes[0].set_xlabel("Domain", fontsize=16)
axes[0].tick_params(axis='x', rotation=45, labelsize=14)
axes[0].tick_params(axis='y', labelsize=14)
axes[0].grid(True, alpha=0.3, axis='y')

# Add domain-specific background colors
for i, domain in enumerate(decoder_domain_success.index):
    if domain in domain_colors:
        axes[0].axvspan(i-0.5, i+0.5, facecolor=domain_colors[domain], alpha=1.0, zorder=0)

# Panel 1b: All Decoders - Generalization Index
if 'Generalization_Index_0to1' in df.columns:
    decoder_domain_gen = df_domain.groupby(['Domain', 'DecoderFamily'])['Generalization_Index_0to1'].mean().unstack(fill_value=0)
    decoder_domain_gen.plot(kind='bar', ax=axes[1], width=0.8,
                           color=sns.color_palette("Set2", len(decoder_domain_gen.columns)))
    axes[1].set_title("(b) Decoder Generalization", fontsize=18, fontweight='bold')
    axes[1].set_ylabel("Generalization Index", fontsize=16)
    axes[1].set_xlabel("Domain", fontsize=16)
    axes[1].tick_params(axis='x', rotation=45, labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
    # Shared legend for panels a and b - positioned between them
    axes[1].legend(title="Decoder Family", fontsize=13, title_fontsize=14, 
                  loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.95)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add domain-specific background colors
    for i, domain in enumerate(decoder_domain_gen.index):
        if domain in domain_colors:
            axes[1].axvspan(i-0.5, i+0.5, facecolor=domain_colors[domain], alpha=1.0, zorder=0)

# Panel 1c: Diffusion vs Other - Success Rate
diffusion_domain_success = df_domain.groupby(['Domain', 'IsDiffusion'])['Adjusted_Success_0to1'].mean().unstack(fill_value=0)
diffusion_domain_success.plot(kind='bar', ax=axes[2], width=0.7,
                              color=['#FF6B6B', '#4ECDC4'], legend=False)
axes[2].set_title("(c) Diffusion vs Other Success", fontsize=18, fontweight='bold')
axes[2].set_ylabel("Normalized Success", fontsize=16)
axes[2].set_xlabel("Domain", fontsize=16)
axes[2].tick_params(axis='x', rotation=45, labelsize=14)
axes[2].tick_params(axis='y', labelsize=14)
axes[2].grid(True, alpha=0.3, axis='y')

# Add domain-specific background colors
for i, domain in enumerate(diffusion_domain_success.index):
    if domain in domain_colors:
        axes[2].axvspan(i-0.5, i+0.5, facecolor=domain_colors[domain], alpha=1.0, zorder=0)

# Panel 1d: Diffusion vs Other - Generalization Index
if 'Generalization_Index_0to1' in df.columns:
    diffusion_domain_gen = df_domain.groupby(['Domain', 'IsDiffusion'])['Generalization_Index_0to1'].mean().unstack(fill_value=0)
    diffusion_domain_gen.plot(kind='bar', ax=axes[3], width=0.7,
                             color=['#FF6B6B', '#4ECDC4'])
    axes[3].set_title("(d) Diffusion vs Other Generalization", fontsize=18, fontweight='bold')
    axes[3].set_ylabel("Generalization Index", fontsize=16)
    axes[3].set_xlabel("Domain", fontsize=16)
    axes[3].tick_params(axis='x', rotation=45, labelsize=14)
    axes[3].tick_params(axis='y', labelsize=14)
    # Shared legend for panels c and d - positioned at the end
    axes[3].legend(title="Decoder Type", fontsize=13, title_fontsize=14, 
                  loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.95)
    axes[3].grid(True, alpha=0.3, axis='y')
    
    # Add domain-specific background colors
    for i, domain in enumerate(diffusion_domain_gen.index):
        if domain in domain_colors:
            axes[3].axvspan(i-0.5, i+0.5, facecolor=domain_colors[domain], alpha=1.0, zorder=0)

plt.tight_layout()
plt.savefig("Plot_script/plots/domain_component_analysis_4panel.svg", dpi=400, bbox_inches='tight')
plt.savefig("Plot_script/plots/domain_component_analysis_4panel.png", dpi=400, bbox_inches='tight')
print("[✓] Domain-wise component analysis saved: Plot_script/plots/domain_component_analysis_4panel.png/svg")
#plt.show()

# ---------------------------------------------------------------------
# 11c. Domain-wise Encoder Analysis: Faceted Plots (Separate subplot per domain)
# ---------------------------------------------------------------------
print("\n[✓] Generating domain-wise encoder analysis (faceted by domain)...")

# Function to group language models into simplified categories
def simplify_language_model(lang_text):
    """Group language models into main categories with short names"""
    lang_text_lower = str(lang_text).lower()
    
    # CLIP-based text encoders
    if 'clip' in lang_text_lower and 'text' in lang_text_lower:
        return 'CLIP Text'
    
    # LLaMA family (LLaMA, Llama, Vicuna which is fine-tuned LLaMA)
    elif 'llama' in lang_text_lower or 'vicuna' in lang_text_lower:
        return 'LLaMA'
    
    # Qwen family (Qwen, Qwen2, Qwen2-VL, etc.)
    elif 'qwen' in lang_text_lower:
        return 'Qwen'
    
    # Gemma/Gemini/PaLI family (PaliGemma, Gemini, PaLI-X, etc.)
    elif 'gemma' in lang_text_lower or 'gemini' in lang_text_lower or 'pali' in lang_text_lower or 'palm' in lang_text_lower:
        return 'Gemma/Gemini'
    
    # GPT family (GPT-3, GPT-4, GPT-4o)
    elif 'gpt' in lang_text_lower:
        return 'GPT'
    
    # T5 family
    elif 't5' in lang_text_lower:
        return 'T5'
    
    # BERT family
    elif 'bert' in lang_text_lower:
        return 'BERT'
    
    # Phi family
    elif 'phi' in lang_text_lower:
        return 'Phi'
    
    # Everything else goes to "Custom Transformers"
    else:
        return 'Custom Transformers'

# Function to simplify vision encoders only for navigation domain
def simplify_navigation_vision_encoder(vision_text):
    """Group navigation vision encoders into 5 categories"""
    vision_text_lower = str(vision_text).lower()
    
    # CLIP-based (CLIP ViT, EVA-CLIP, Molmo-CLIP, fusions with CLIP)
    if 'clip' in vision_text_lower:
        return 'CLIP-based'
    
    # DINOv2
    elif 'dino' in vision_text_lower:
        return 'DINOv2'
    
    # ResNet
    elif 'resnet' in vision_text_lower:
        return 'ResNet'
    
    # Qwen Vision
    elif 'qwen' in vision_text_lower:
        return 'Qwen Vision'
    
    # Custom ViT (everything else including Gemini ViT, Phi Vision, ViT+LoRA, VILA, OpenVLA, EVA-02, etc.)
    else:
        return 'Custom ViT'

# For the faceted plot, use actual encoder names for vision but grouped categories for language
# Prepare vision encoder data using actual Vision_Encoder names
vision_domain_data_individual = df_domain.groupby(['Domain', 'Vision_Encoder'])['Adjusted_Success_0to1'].mean().reset_index()
vision_domain_data_individual = vision_domain_data_individual[vision_domain_data_individual['Adjusted_Success_0to1'] > 0]

# Apply simplification only for navigation domain vision encoders
nav_vision_simplified = df_domain[df_domain['Domain'] == 'navigation'].copy()
nav_vision_simplified['Vision_Encoder_Simplified'] = nav_vision_simplified['Vision_Encoder'].apply(simplify_navigation_vision_encoder)
nav_vision_grouped = nav_vision_simplified.groupby(['Domain', 'Vision_Encoder_Simplified'])['Adjusted_Success_0to1'].mean().reset_index()
nav_vision_grouped = nav_vision_grouped[nav_vision_grouped['Adjusted_Success_0to1'] > 0]
nav_vision_grouped.rename(columns={'Vision_Encoder_Simplified': 'Vision_Encoder'}, inplace=True)

# Replace navigation vision data with simplified version
vision_domain_data_individual = vision_domain_data_individual[vision_domain_data_individual['Domain'] != 'navigation']
vision_domain_data_individual = pd.concat([vision_domain_data_individual, nav_vision_grouped], ignore_index=True)

# Prepare language model data using simplified categories
df_domain['Language_Simplified'] = df_domain['Language_Encoder'].apply(simplify_language_model)
lang_domain_data_individual = df_domain.groupby(['Domain', 'Language_Simplified'])['Adjusted_Success_0to1'].mean().reset_index()
lang_domain_data_individual = lang_domain_data_individual[lang_domain_data_individual['Adjusted_Success_0to1'] > 0]
# Rename column for consistency
lang_domain_data_individual.rename(columns={'Language_Simplified': 'Language_Encoder'}, inplace=True)

# For manipulation domain: only keep encoders that appear at least twice (vision only, keep all language groups)
manipulation_vision_counts = df_domain[df_domain['Domain'] == 'manipulation']['Vision_Encoder'].value_counts()
manipulation_vision_keep = manipulation_vision_counts[manipulation_vision_counts >= 2].index.tolist()

# Filter manipulation vision data only
vision_domain_data_individual = vision_domain_data_individual[
    (vision_domain_data_individual['Domain'] != 'manipulation') | 
    (vision_domain_data_individual['Vision_Encoder'].isin(manipulation_vision_keep))
]

# Language data is already grouped, so no need to filter by occurrence count

# Get domains
domains = df_domain['Domain'].unique()
n_domains = len(domains)

# Create figure with 2 rows (vision encoders, language models) x n_domains columns
fig, axes = plt.subplots(2, n_domains, figsize=(8*n_domains, 14))
if n_domains == 1:
    axes = axes.reshape(-1, 1)

fig.suptitle("Domain-specific Encoder Performance Analysis", 
             fontsize=24, fontweight='bold', y=0.98)

# Row 1: Vision Encoders by Domain (individual encoders)
for idx, domain in enumerate(sorted(domains)):
    domain_vision = vision_domain_data_individual[vision_domain_data_individual['Domain'] == domain].sort_values('Adjusted_Success_0to1', ascending=True)
    
    if len(domain_vision) > 0:
        y_pos = range(len(domain_vision))
        axes[0, idx].barh(y_pos, domain_vision['Adjusted_Success_0to1'], 
                         color=sns.color_palette("pastel", len(domain_vision)),
                         alpha=0.9, edgecolor='gray', linewidth=0.8)
        axes[0, idx].set_yticks(y_pos)
        # Truncate long encoder names for readability
        encoder_labels = [enc[:40] + '...' if len(enc) > 40 else enc for enc in domain_vision['Vision_Encoder']]
        axes[0, idx].set_yticklabels(encoder_labels, fontsize=14)
        axes[0, idx].set_xlabel('Normalized Success', fontsize=16, fontweight='bold')
        
        # Increase x-axis tick label size
        axes[0, idx].tick_params(axis='x', labelsize=13)
        
        # Clean title without counts or filtering info
        axes[0, idx].set_title(f"{domain.capitalize()}\nVision Encoders", 
                              fontsize=18, fontweight='bold', pad=10)
        
        axes[0, idx].set_xlim(0, 1.1)
        axes[0, idx].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (y, val) in enumerate(zip(y_pos, domain_vision['Adjusted_Success_0to1'])):
            axes[0, idx].text(val + 0.02, y, f'{val:.3f}', va='center', fontsize=12)
        
        # No background color
    else:
        axes[0, idx].text(0.5, 0.5, f'No vision encoder data\nfor {domain}', 
                         ha='center', va='center', fontsize=12, transform=axes[0, idx].transAxes)
        axes[0, idx].set_xlim(0, 1)

# Row 2: Language Models by Domain (individual language encoders)
for idx, domain in enumerate(sorted(domains)):
    domain_lang = lang_domain_data_individual[lang_domain_data_individual['Domain'] == domain].sort_values('Adjusted_Success_0to1', ascending=True)
    
    if len(domain_lang) > 0:
        y_pos = range(len(domain_lang))
        axes[1, idx].barh(y_pos, domain_lang['Adjusted_Success_0to1'],
                         color=sns.color_palette("pastel", len(domain_lang)),
                         alpha=0.9, edgecolor='gray', linewidth=0.8)
        axes[1, idx].set_yticks(y_pos)
        # Truncate long language model names for readability
        lang_labels = [lang[:40] + '...' if len(lang) > 40 else lang for lang in domain_lang['Language_Encoder']]
        axes[1, idx].set_yticklabels(lang_labels, fontsize=14)
        axes[1, idx].set_xlabel('Normalized Success', fontsize=16, fontweight='bold')
        
        # Increase x-axis tick label size
        axes[1, idx].tick_params(axis='x', labelsize=13)
        
        # Clean title without counts or filtering info
        axes[1, idx].set_title(f"{domain.capitalize()}\nLanguage Models", 
                              fontsize=18, fontweight='bold', pad=10)
        
        axes[1, idx].set_xlim(0, 1.1)
        axes[1, idx].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (y, val) in enumerate(zip(y_pos, domain_lang['Adjusted_Success_0to1'])):
            axes[1, idx].text(val + 0.02, y, f'{val:.3f}', va='center', fontsize=12)
        
        # No background color
    else:
        axes[1, idx].text(0.5, 0.5, f'No language model data\nfor {domain}', 
                         ha='center', va='center', fontsize=12, transform=axes[1, idx].transAxes)
        axes[1, idx].set_xlim(0, 1)

plt.tight_layout()
plt.savefig("Plot_script/plots/encoder_domain_faceted.svg", dpi=400, bbox_inches='tight')
plt.savefig("Plot_script/plots/encoder_domain_faceted.png", dpi=400, bbox_inches='tight')
print("[✓] Domain-wise encoder analysis (faceted) saved: Plot_script/plots/encoder_domain_faceted.png/svg")
#plt.show()

# ---------------------------------------------------------------------
# 11d. Domain-wise Encoder Analysis: Grouped Horizontal Bars
# ---------------------------------------------------------------------
# 12. Merged Decoder and Encoder Analysis: 6-Panel (2x3 layout)
# ---------------------------------------------------------------------
print("\n[✓] Generating merged 6-panel decoder and encoder analysis...")

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle("Comprehensive Component Performance Analysis: Decoders, Vision Encoders, and Language Models", 
             fontsize=20, fontweight='bold', y=0.98)

# Row 1: Decoder Analysis (2 panels)
# Panel 1a: Normalized Success by Decoder Family
decoder_success = df.groupby('DecoderFamily')['Adjusted_Success_0to1'].agg(['mean', 'std', 'count']).reset_index()
decoder_success = decoder_success[decoder_success['DecoderFamily'] != 'flow']  # Remove "flow" (only 2 models)
decoder_success = decoder_success.sort_values('mean', ascending=False)

bars1 = axes[0,0].bar(decoder_success['DecoderFamily'], decoder_success['mean'], 
                    yerr=decoder_success['std'], capsize=5,
                    color=sns.color_palette("rocket", len(decoder_success)),
                    alpha=0.8, edgecolor='black')
axes[0,0].set_title("(a) Success Rate by Decoder Family", fontsize=16, fontweight='bold')
axes[0,0].set_ylabel("Normalized Success", fontsize=14)
axes[0,0].set_xlabel("Decoder Family", fontsize=14)
axes[0,0].tick_params(axis='x', rotation=45, labelsize=12)
axes[0,0].tick_params(axis='y', labelsize=12)
axes[0,0].grid(True, alpha=0.3)

# Add value labels
for i, bar in enumerate(bars1):
    height = bar.get_height()
    axes[0,0].text(bar.get_x() + 0.05, 0.02,
                f'{height:.3f}', ha='left', va='bottom', fontweight='bold', fontsize=11)

# Panel 1b: Generalization Index by Decoder Family
if 'Generalization_Index_0to1' in df.columns:
    decoder_gen = df.groupby('DecoderFamily')['Generalization_Index_0to1'].agg(['mean', 'std', 'count']).reset_index()
    decoder_gen = decoder_gen[decoder_gen['DecoderFamily'] != 'flow']  # Remove "flow" (only 2 models)
    decoder_gen = decoder_gen.sort_values('mean', ascending=False)
    
    bars2 = axes[0,1].bar(decoder_gen['DecoderFamily'], decoder_gen['mean'], 
                        yerr=decoder_gen['std'], capsize=5,
                        color=sns.color_palette("mako", len(decoder_gen)),
                        alpha=0.8, edgecolor='black')
    axes[0,1].set_title("(b) Generalization Index by Decoder Family", fontsize=16, fontweight='bold')
    axes[0,1].set_ylabel("Generalization Index", fontsize=14)
    axes[0,1].set_xlabel("Decoder Family", fontsize=14)
    axes[0,1].tick_params(axis='x', rotation=45, labelsize=12)
    axes[0,1].tick_params(axis='y', labelsize=12)
    axes[0,1].grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        axes[0,1].text(bar.get_x() + 0.05, 0.02,
                    f'{height:.3f}', ha='left', va='bottom', fontweight='bold', fontsize=11)

# Panel 1c: Vision Encoder Success Rate
vision_success = df.groupby('VisionEncoderFamily')['Adjusted_Success_0to1'].agg(['mean', 'std', 'count']).reset_index()
vision_success = vision_success[vision_success['VisionEncoderFamily'] != 'Other']  # Remove "Other"
vision_success = vision_success[vision_success['count'] >= 3]  # Only include families with n >= 3
vision_success = vision_success.sort_values('mean', ascending=False)

bars3 = axes[0,2].bar(vision_success['VisionEncoderFamily'], vision_success['mean'], 
                    yerr=vision_success['std'], capsize=5,
                    color=sns.color_palette("viridis", len(vision_success)),
                    alpha=0.8, edgecolor='black')
axes[0,2].set_title("(c) Success Rate by Vision Encoder Family", fontsize=16, fontweight='bold')
axes[0,2].set_ylabel("Normalized Success", fontsize=14)
axes[0,2].set_xlabel("Vision Encoder Family", fontsize=14)
axes[0,2].tick_params(axis='x', rotation=45, labelsize=12)
axes[0,2].tick_params(axis='y', labelsize=12)
axes[0,2].grid(True, alpha=0.3)

# Add value labels
for i, bar in enumerate(bars3):
    height = bar.get_height()
    axes[0,2].text(bar.get_x() + 0.05, 0.02,
                  f'{height:.3f}', ha='left', va='bottom', fontweight='bold', fontsize=11)

# Row 2: Encoder Analysis continuation (3 panels)
# Panel 2a: Vision Encoder Generalization Index
if 'Generalization_Index_0to1' in df.columns:
    vision_gen = df.groupby('VisionEncoderFamily')['Generalization_Index_0to1'].agg(['mean', 'std', 'count']).reset_index()
    vision_gen = vision_gen[vision_gen['VisionEncoderFamily'] != 'Other']  # Remove "Other"
    vision_gen = vision_gen[vision_gen['count'] >= 3]  # Only include families with n >= 3
    vision_gen = vision_gen.sort_values('mean', ascending=False)
    
    bars4 = axes[1,0].bar(vision_gen['VisionEncoderFamily'], vision_gen['mean'], 
                        yerr=vision_gen['std'], capsize=5,
                        color=sns.color_palette("cividis", len(vision_gen)),
                        alpha=0.8, edgecolor='black')
    axes[1,0].set_title("(d) Generalization Index by Vision Encoder", fontsize=16, fontweight='bold')
    axes[1,0].set_ylabel("Generalization Index", fontsize=14)
    axes[1,0].set_xlabel("Vision Encoder Family", fontsize=14)
    axes[1,0].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1,0].tick_params(axis='y', labelsize=12)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + 0.05, 0.02,
                      f'{height:.3f}', ha='left', va='bottom', fontweight='bold', fontsize=11)

# Panel 2b: Language Model Success Rate
lang_success = df.groupby('LanguageModelFamily')['Adjusted_Success_0to1'].agg(['mean', 'std', 'count']).reset_index()
lang_success = lang_success[~lang_success['LanguageModelFamily'].isin(['Other', 'None/Other'])]  # Remove "Other" and "None/Other"
lang_success = lang_success[lang_success['count'] >= 3]  # Only include families with n >= 3
lang_success = lang_success.sort_values('mean', ascending=False)

bars5 = axes[1,1].bar(lang_success['LanguageModelFamily'], lang_success['mean'], 
                    yerr=lang_success['std'], capsize=5,
                    color=sns.color_palette("plasma", len(lang_success)),
                    alpha=0.8, edgecolor='black')
axes[1,1].set_title("(e) Success Rate by Language Model Family", fontsize=16, fontweight='bold')
axes[1,1].set_ylabel("Normalized Success", fontsize=14)
axes[1,1].set_xlabel("Language Model Family", fontsize=14)
axes[1,1].tick_params(axis='x', rotation=45, labelsize=12)
axes[1,1].tick_params(axis='y', labelsize=12)
axes[1,1].grid(True, alpha=0.3)

# Add value labels
for i, bar in enumerate(bars5):
    height = bar.get_height()
    axes[1,1].text(bar.get_x() + 0.05, 0.02,
                  f'{height:.3f}', ha='left', va='bottom', fontweight='bold', fontsize=11)

# Panel 2c: Language Model Generalization Index
if 'Generalization_Index_0to1' in df.columns:
    lang_gen = df.groupby('LanguageModelFamily')['Generalization_Index_0to1'].agg(['mean', 'std', 'count']).reset_index()
    lang_gen = lang_gen[~lang_gen['LanguageModelFamily'].isin(['Other', 'None/Other'])]  # Remove "Other" and "None/Other"
    lang_gen = lang_gen[lang_gen['count'] >= 3]  # Only include families with n >= 3
    lang_gen = lang_gen.sort_values('mean', ascending=False)
    
    bars6 = axes[1,2].bar(lang_gen['LanguageModelFamily'], lang_gen['mean'], 
                        yerr=lang_gen['std'], capsize=5,
                        color=sns.color_palette("magma", len(lang_gen)),
                        alpha=0.8, edgecolor='black')
    axes[1,2].set_title("(f) Generalization Index by Language Model", fontsize=16, fontweight='bold')
    axes[1,2].set_ylabel("Generalization Index", fontsize=14)
    axes[1,2].set_xlabel("Language Model Family", fontsize=14)
    axes[1,2].tick_params(axis='x', rotation=45, labelsize=12)
    axes[1,2].tick_params(axis='y', labelsize=12)
    axes[1,2].grid(True, alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars6):
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + 0.05, 0.02,
                      f'{height:.3f}', ha='left', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig("Plot_script/plots/merged_decoder_encoder_6panel.svg", dpi=300, bbox_inches='tight')
plt.savefig("Plot_script/plots/merged_decoder_encoder_6panel.png", dpi=300, bbox_inches='tight')
print("[✓] Merged 6-panel decoder and encoder analysis saved: Plot_script/plots/merged_decoder_encoder_6panel.png/svg")
#plt.show()

# =======================================================
# VLA-Fusion Theoretical Quantities Visualization (3-Panel)
# =======================================================
print("\n[✓] Generating VLA fusion theory 3-panel visualization...")

# Compute theoretical quantities using original (unstandardized) values
df["DeltaH"] = 1 - df["Adjusted_Success_0to1"]
df["Eta"] = (df["CMod_orig"] * df["CTask_orig"]) / (df["VisionParams_orig"] + df["LLMParams_orig"])
df["Comp"] = df["CMod_orig"] - df["Difficulty_Index_orig"]
df["E_fusion"] = (
    df["Adjusted_Success_0to1"]
    * (df["VisionParams_orig"] + df["LLMParams_orig"])
    * np.log1p(df["CMod_orig"])
)

# Ensure FusionDepth is treated as categorical with proper ordering
order = ["early", "late", "hierarch"]
df["FusionDepth_ordered"] = pd.Categorical(df["FusionDepth_cat"], categories=order, ordered=True)

# Create 3-panel figure
sns.set(style="whitegrid", font_scale=1.3)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ----- (a) Entropy Reduction -----
ax1 = axes[0]
sns.boxplot(data=df, x="FusionDepth_ordered", y="DeltaH", palette="pastel", ax=ax1)
ax1.set_title("(a) Entropy Reduction Across Fusion Depths", weight="bold", fontsize=14, pad=12)
ax1.set_xlabel("Fusion Depth Category", fontsize=13, labelpad=10)
ax1.set_ylabel("ΔHₖ (Entropy Reduction)", fontsize=13, labelpad=10)
ax1.tick_params(axis='both', labelsize=12)
ax1.grid(True, alpha=0.3, axis='y')

# ----- (b) Cross-Modal Attention Efficiency -----
ax2 = axes[1]

# Light, refined pastel color palette (3 colors for early, late, hierarch)
box_palette = sns.color_palette(["#87CEEB", "#98D8C8", "#DDA0DD"])

# --- Box plot for clear statistical representation ---
sns.boxplot(
    data=df,
    x="FusionDepth_ordered",
    y="Eta",
    order=order,
    palette=box_palette,
    width=0.6,
    linewidth=1.5,
    hue="FusionDepth_ordered",
    legend=False,
    ax=ax2
)

ax2.set_title("(b) Cross-Modal Attention Efficiency", 
              weight="bold", fontsize=14, pad=12)
ax2.set_xlabel("Fusion Depth Category", fontsize=13, labelpad=10)
ax2.set_ylabel("ηₖ (Attention Efficiency)", fontsize=13, labelpad=10)
ax2.set_ylim(-0.004, 0.075)
ax2.tick_params(axis='both', labelsize=12)
ax2.grid(True, alpha=0.3, axis='y')

# ----- (c) Fusion Energy vs Success -----
ax3 = axes[2]
from scipy.stats import pearsonr
sns.regplot(
    data=df,
    x="E_fusion",
    y="Adjusted_Success_0to1",
    scatter_kws={'s': 60, 'alpha': 0.7},
    color="teal",
    ax=ax3
)
r, p = pearsonr(df["E_fusion"], df["Adjusted_Success_0to1"])
ax3.set_title("(c) Fusion Energy vs Success", 
              weight="bold", fontsize=14, pad=12)
ax3.set_xlabel("Fusion Energy", fontsize=13, labelpad=10)
ax3.set_ylabel("Normalized Task Success", fontsize=13, labelpad=10)
ax3.tick_params(axis='both', labelsize=12)
ax3.grid(True, alpha=0.3)


plt.tight_layout()
plt.savefig("Plot_script/plots/vla_fusion_theory_visualization_3panel.pdf", bbox_inches="tight", dpi=300)
plt.savefig("Plot_script/plots/vla_fusion_theory_visualization_3panel.png", dpi=600, bbox_inches="tight")
print("[✓] VLA fusion theory 3-panel visualization saved: Plot_script/plots/vla_fusion_theory_visualization_3panel.png/pdf")
#plt.show()


# ---------------------------------------------------------------------
# VLA-FEB Histogram Bar Chart
# ---------------------------------------------------------------------
print("\n[✓] Generating VLA-FEB Histogram Bar Chart...")

# Load top75.csv with VLA-FEB components
df_feb = pd.read_csv("top75.csv")

# Filter out specific models
models_to_remove = ["Gato", "Shake-VLA", "SaM2ACT", "SAM2ACT", "SAM2Act", "OpenVLA-OFT", "VLATest"]
df_feb = df_feb[~df_feb["Model"].isin(models_to_remove)]

# Define weights (equal weights for all components)
w_fusion = 0.25  # Fusion Energy weight
w_gi = 0.25      # Generalization Index weight
w_r2s = 0.25     # Robustness-to-Success weight
w_cmas = 0.25    # Combined Model Architecture Scale weight

# Compute VLA_FEB_Score with custom weights
df_feb["VLA_FEB_Score_Custom"] = (
    w_cmas * df_feb["CMAS"] +
    w_fusion * df_feb["E_fusion"] +
    w_r2s * df_feb["R2S"] +
    w_gi * df_feb["GI_actual"]
)

# Sort dataframe by VLA_FEB_Score_Custom in descending order for bar chart
df_feb_sorted = df_feb.sort_values("VLA_FEB_Score_Custom", ascending=False)

plt.figure(figsize=(14, 6))
plt.bar(df_feb_sorted["Model"], df_feb_sorted["VLA_FEB_Score_Custom"], 
        color="mediumseagreen", alpha=0.85, edgecolor='darkgreen', linewidth=0.8)
plt.xticks(rotation=60, ha="right", fontsize=10)
plt.ylabel("VLA-FEB Composite Score", fontsize=13, fontweight='bold')
plt.title(f"VLA-FEB Composite Scores (Equal Weights: w={w_fusion})", 
          fontsize=15, fontweight='bold', pad=15)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.ylim(0, max(df_feb_sorted["VLA_FEB_Score_Custom"]) * 1.1)
plt.tight_layout()

# Save to plots directory
plt.savefig("plots/VLA_FEB_Hist.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"[✓] VLA-FEB Histogram saved: plots/VLA_FEB_Hist.png")
print(f"    Models included: {len(df_feb_sorted)}")
top3 = ', '.join(df_feb_sorted.head(3)['Model'].tolist())
print(f"    Top 3 models: {top3}")







