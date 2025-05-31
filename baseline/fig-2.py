import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the chart
models = ['DeepSeek R1', 'Grok 3', 'o4-mini', 'Claude-3.7-sonnet', 'gemini-2.5-flash']

# Performance data (in percentages)
explanation_consistency = [91.7, 96.0, 90.7, 91.1, 92.3]
request_comprehension = [61.5, 67.9, 62.2, 59.1, 64.0]
semantic_equivalence = [86.4, 93.7, 93.0, 36.0, 93.5]

# Set up the plot with publication-ready styling
plt.style.use('default')
fig, ax = plt.subplots(figsize=(12, 8))

# Set positions for bars
x = np.arange(len(models))
width = 0.25

# Create bars with colors matching the original
bars1 = ax.bar(x - width, explanation_consistency, width, 
               label='Explanation Consistency', color='#2E5984', alpha=0.9)
bars2 = ax.bar(x, request_comprehension, width, 
               label='Request Comprehension', color='#B85450', alpha=0.9)
bars3 = ax.bar(x + width, semantic_equivalence, width, 
               label='Semantic Equivalence', color='#4A7C59', alpha=0.9)

# Customize the plot for publication
ax.set_xlabel('Model', fontsize=16, fontweight='bold')
ax.set_ylabel('Performance (%)', fontsize=16, fontweight='bold')

# Set x-axis labels
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=14)

# Set y-axis range and ticks
ax.set_ylim(30, 100)
ax.set_yticks(range(30, 101, 10))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

# Add value labels on top of each bar
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=12, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

# Move legend to top center and enlarge font
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3,
          fontsize=14, frameon=True, fancybox=True, shadow=True)

# Add grid for better readability
ax.grid(True, linestyle='--', alpha=0.7, axis='y')
ax.set_axisbelow(True)

# Improve tick formatting
ax.tick_params(axis='both', which='major', labelsize=13)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# For papers, you might want to save as high-resolution PNG or PDF
# Uncomment the line below to save the figure
# plt.savefig('model_performance_comparison.pdf', dpi=300, bbox_inches='tight')

plt.show()