import numpy as np
import matplotlib.pyplot as plt
import os
from grid_env import GridWorld
from mc_agent import MCAgent

def plot_results(agent, eps, save_path):
    policy_map, value_map = agent.get_optimal_policy_map()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, agent.env.cols)
    ax.set_ylim(agent.env.rows, 0)
    ax.set_xticks(np.arange(0, agent.env.cols + 1))
    ax.set_yticks(np.arange(0, agent.env.rows + 1))
    ax.grid(True)
    
    # Draw grid cells
    for r in range(agent.env.rows):
        for c in range(agent.env.cols):
            # Background color
            color = 'white'
            if (r, c) in agent.env.forbidden_states:
                color = '#ffcc99' # Light orange
            elif (r, c) == agent.env.target_state:
                color = '#99ccff' # Light blue
            
            rect = plt.Rectangle((c, r), 1, 1, facecolor=color, edgecolor='gray')
            ax.add_patch(rect)
            
            # Draw Value
            val = value_map[r, c]
            ax.text(c + 0.5, r + 0.5, f"{val:.2f}", ha='center', va='center', fontsize=10, color='black')
            
            # Draw Arrow for Policy
            action = policy_map[r, c]
            # 0:Up, 1:Right, 2:Down, 3:Left, 4:Stay
            dx, dy = 0, 0
            if action == 0: dy = -0.3
            elif action == 1: dx = 0.3
            elif action == 2: dy = 0.3
            elif action == 3: dx = -0.3
            
            if action != 4:
                ax.arrow(c + 0.5, r + 0.5, dx, dy, head_width=0.1, head_length=0.1, fc='k', ec='k')
            else:
                ax.plot(c + 0.5, r + 0.5, 'ko', markersize=5)

    plt.title(f"Optimal Policy & Values (eps={eps})")
    plt.savefig(save_path)
    plt.close()

def run_experiments():
    if not os.path.exists('results_src1'):
        os.makedirs('results_src1')

    epsilons = [0, 0.1, 0.2, 0.5]
    num_episodes = 20000 # Reduced as requested
    
    for eps in epsilons:
        print(f"Running experiment for epsilon={eps}...")
        env = GridWorld()
        # Increased alpha to 0.05 for faster convergence within 20k episodes
        # With 20k episodes and ES, each (s,a) is visited ~160 times.
        # alpha=0.05 allows significant updates.
        agent = MCAgent(env, epsilon=eps, gamma=0.9, alpha=0.05)
        
        for i in range(num_episodes):
            # Use Exploring Starts to ensure all states are visited
            episode = agent.generate_episode(max_steps=200, exploring_starts=True)
            agent.update(episode)
            
            if (i+1) % 5000 == 0:
                print(f"  Episode {i+1}/{num_episodes}")
        
        # Plot
        plot_results(agent, eps, f"results_src1/policy_eps_{eps}.png")
        print(f"Saved results for epsilon={eps}")

if __name__ == "__main__":
    run_experiments()
