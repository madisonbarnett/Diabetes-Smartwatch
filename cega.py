import matplotlib.pyplot as plt
import numpy as np

def get_cega_zone(ref, pred):
    # Store actual (ref) and predicted (pred) vals as floats
    r, p = float(ref), float(pred)

    # Classify points into zones
    if (r <= 70 and p <= 70) or (0.8*r <= p <= 1.2*r):
        return 'A'

    if (130 < r <= 180 and 1.4*(r-130) >= p) or (70 < r <= 280 and p >= (r+110)):
        return 'C'

    if (r <= 70 and 70 < p <= 180) or (r >= 240 and 70 <= p <= 180):
        return 'D'

    if (r <= 70 and p > 180) or (r > 180 and p <= 70):
        return 'E'

    return 'B'      # B is defined as not in any other zone (complex logic, easier this way)

def cega(y_orig_test, y_pred):
    zones_count = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}
    total_points = len(y_orig_test)
    points = []  # Store (ref, pred, zone) for plotting

    for ref, pred in zip(y_orig_test, y_pred):
        zone = get_cega_zone(ref, pred)     # Classify point into zone
        zones_count[zone] += 1              # Increment zone count
        points.append((ref, pred, zone))    # Store point

    # Print zone statistics
    print("Clarke Error Grid Analysis:")
    for zone, count in zones_count.items():
        percentage = (count / total_points) * 100
        print(f"Zone {zone}: {percentage:.2f}% ({count}/{total_points} points)")

    # Create plot
    plt.figure(figsize=(10, 10))

    # Define zone colors and labels
    colors = {'A': 'green', 'B': 'yellow', 'C': 'orange', 'D': 'red', 'E': 'purple'}
    labels = {
        'A': 'A: Clinically Accurate',
        'B': 'B: Benign Errors',
        'C': 'C: Overcorrection',
        'D': 'D: Dangerous Failure to Detect',
        'E': 'E: Erroneous Treatment'
    }

    # Scatter points by zone
    for zone in 'ABCDE':
        zone_points = [(r, p) for r, p, z in points if z == zone]   # Plot points in valid zones
        if zone_points:
            refs, preds = zip(*zone_points)
            plt.scatter(refs, preds, c=colors[zone], label=f"{labels[zone]} ({zones_count[zone]})", alpha=0.7, edgecolors='k', s=60)

    # Draw grid boundaries
    max_val = 400
    x = np.linspace(0, max_val, 500)

    # Perfect line (y = x)
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='Perfect Agreement')

    # Zone A boundaries: +/- 20% or within 70
    plt.fill_between(x, 0.8*x, 1.2*x, where=(x <= 70) | (x >= 70), color='green', alpha=0.1, label='_nolegend_')
    plt.fill_between(x, 0, 70, where=x <= 70, color='green', alpha=0.1)

    # Zone B: outside A but safe
    # Complex boundaries, draw other zones instead

    # Zone C: 
    plt.fill([70, 70, 290], [180, 400, 400], color='orange', alpha=0.4)
    plt.fill([130, 180, 180], [0, 0, 70], color='orange', alpha=0.4)

    # Zone D: 
    # x: left, right, right, left
    # y: bottom, bottom, top, top

    plt.fill([0, 70, 70, 0], [70, 70, 180, 180], color='red', alpha=0.1)
    plt.fill([240, 400, 400, 240], [70, 70, 180, 180], color='red', alpha=0.1)

    # Zone E: 

    plt.fill([0, 70, 70, 0], [180, 180, 400, 400], color='purple', alpha=0.1)
    plt.fill([180, 400, 400, 180], [0, 0, 70, 70], color='purple', alpha=0.1)

    # Axis limits and labels
    plt.xlim(0, max_val)
    plt.ylim(0, max_val)
    plt.xlabel('Reference Glucose (mg/dL)', fontsize=12)    # Actual
    plt.ylabel('Predicted Glucose (mg/dL)', fontsize=12)    # Predicted
    plt.title('Clarke Error Grid Analysis', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='box')

    # Legend
    plt.legend(loc='upper left')

    # Show plot
    plt.tight_layout()
    plt.show()