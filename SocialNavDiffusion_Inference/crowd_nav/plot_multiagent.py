import matplotlib.pyplot as plt

# -----------------------------
# INPUT DATA (edit as needed)
# -----------------------------
planners = [
    "Single-Agent Diffusion",
    "Multi-Agent Diffusion",
    "ORCA",
    "Social Force",
    "CADRL",
    "SARL",
    "LSTM-RL"
]

# Success rates in percentage
success_rates = [
    97.5,
    99,
    94.5,
    95.8,
    86.8,
    85.5,
    86
]

# -----------------------------
# CREATE BAR GRAPH
# -----------------------------
plt.figure(figsize=(8, 5))

plt.bar(planners, success_rates)

# Labels and title
plt.ylabel("Success Rate (%)")
plt.xlabel("Planner")
plt.title("Planner Success Rate Comparison")

# Rotate x labels for readability
plt.xticks(rotation=20)

# Add value labels on top of bars
for i, v in enumerate(success_rates):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center')

# Layout adjustment
plt.tight_layout()

# -----------------------------
# SAVE IMAGE
# -----------------------------
output_path = "planner_success_rates.png"
plt.savefig(output_path, dpi=300)

# Optional: display the plot
plt.show()

print(f"Saved plot to {output_path}")