import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_value_function(V, rows=5, cols=5, title="Value Function", save_path=None):
    """
    Plots the value function as a heatmap.
    Args:
        V: Dict or Array of values.
        rows: Grid rows.
        cols: Grid cols.
        save_path: Path to save the figure. If None, show it.
    """
    grid = np.zeros((rows, cols))
    if isinstance(V, dict):
        for r in range(rows):
            for c in range(cols):
                grid[r, c] = V.get((r, c), 0)
    else:
        grid = V.reshape((rows, cols))
        
    plt.figure(figsize=(8, 6))
    sns.heatmap(grid, annot=True, fmt=".2f", cmap="viridis")
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_learning_curve(history, title="Learning Curve", ylabel="Error", save_path=None):
    """
    Plots a learning curve.
    Args:
        history: List of values to plot.
        save_path: Path to save the figure. If None, show it.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history)
    plt.title(title)
    plt.xlabel("Episodes/Iterations")
    plt.ylabel(ylabel)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compare_algorithms(histories, labels, title="Algorithm Comparison", ylabel="Value", save_path=None):
    """
    Compare multiple learning curves.
    """
    plt.figure(figsize=(10, 6))
    for hist, label in zip(histories, labels):
        plt.plot(hist, label=label)
    plt.title(title)
    plt.xlabel("Episodes/Iterations")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_policy(policy, rows=5, cols=5, forbidden_states=None, target_state=None, title="Optimal Policy", save_path=None):
    """
    Plots the policy as a grid with arrows.
    Args:
        policy: Dictionary mapping state (r, c) to action index.
        rows: Grid rows.
        cols: Grid cols.
        forbidden_states: List of forbidden states (r, c).
        target_state: Target state (r, c).
        save_path: Path to save the figure.
    """
    if forbidden_states is None:
        forbidden_states = []
        
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Action mapping for arrows
    # 0: 'up', 1: 'down', 2: 'left', 3: 'right', 4: 'stay'
    action_arrows = {0: '↑', 1: '↓', 2: '←', 3: '→', 4: '•'}
    
    for r in range(rows):
        for c in range(cols):
            state = (r, c)
            
            # Determine background color
            if state in forbidden_states:
                color = '#808080' # Gray
            elif state == target_state:
                color = '#90EE90' # Light Green
            else:
                color = 'white'
            
            # Draw cell
            # Note: In matplotlib, (0,0) is bottom-left. 
            # To match matrix coordinates where (0,0) is top-left:
            # x = c, y = rows - 1 - r
            rect = plt.Rectangle((c, rows - 1 - r), 1, 1, facecolor=color, edgecolor='black')
            ax.add_patch(rect)
            
            # Draw arrow
            # Draw arrow for all states including forbidden and target if policy exists
            action = policy.get(state)
            if action is not None:
                arrow = action_arrows.get(action, '?')
                # If it's target state, maybe combine 'G' with arrow or just show arrow?
                # User asked for action policy on target too.
                # Let's overlay arrow on top of color.
                
                if state == target_state:
                    # Show G and arrow? Or just arrow?
                    # Let's show arrow, but maybe smaller or next to G?
                    # Or just replace G with arrow as requested "禁区应该也要有行动策略终点也是"
                    # But usually target is terminal. If it has policy, it means what to do FROM target.
                    # Let's just print the arrow.
                    ax.text(c + 0.5, rows - 1 - r + 0.5, arrow, 
                           ha='center', va='center', fontsize=20, fontweight='bold')
                else:
                    ax.text(c + 0.5, rows - 1 - r + 0.5, arrow, 
                           ha='center', va='center', fontsize=20, fontweight='bold')
            elif state == target_state:
                 # Fallback if no policy for target
                 ax.text(c + 0.5, rows - 1 - r + 0.5, "G", 
                           ha='center', va='center', fontsize=20, fontweight='bold')
                    
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
