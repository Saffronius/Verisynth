import matplotlib.pyplot as plt
import numpy as np

# Data
string_sizes = [100, 500, 1000, 1500, 2000]
verisynth_scores = [0.824, 0.814, 0.889, 0.717, 0.814]
baseline_score = 0.322  # Only at 1000 string size

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create bar positions
x_pos = np.arange(len(string_sizes))
# Define bar width for tighter spacing
width = 0.6  # width of the bars
half_width = width / 2  # half bar width for split bars
widths = [width] * len(string_sizes)
widths[2] = half_width  # split the 1000 string size bar into two halves

# Plot VERISYNTH bars
bars1 = ax.bar(x_pos, verisynth_scores, width=widths, 
               color='#4A90E2', label='VERISYNTH', alpha=0.8)

# Plot baseline bar only at position 2 (1000 string size)
bars2 = ax.bar(x_pos[2] + half_width, baseline_score, width=half_width, 
               color='#E8743B', label='Z3 + LLM Baseline', alpha=0.8)

# Add value labels on bars
for i, bar in enumerate(bars1):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{verisynth_scores[i]:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Add label for baseline bar
ax.text(bars2[0].get_x() + bars2[0].get_width()/2., baseline_score + 0.01,
        f'{baseline_score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

# Customize the plot
ax.set_xlabel('String Size', fontsize=16, fontweight='bold')
ax.set_ylabel('Jaccard Similarity Score', fontsize=16, fontweight='bold')
ax.set_ylim(0, 1.0)
# Position x-axis labels at the center of each bar or bar group
tick_positions = list(x_pos)
tick_positions[2] = x_pos[2] + half_width/2
ax.set_xticks(tick_positions)
ax.set_xticklabels([f'{size:,}' for size in string_sizes], fontsize=12)
ax.set_title('')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
ax.tick_params(axis='both', labelsize=12)

# Add error bars if you have standard deviations (optional)
# ax.errorbar(x_pos, verisynth_scores, yerr=your_std_values, fmt='none', color='black', capsize=5)

plt.tight_layout()
plt.show()