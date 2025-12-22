def calculate_l_infinity_norm(V1, V2):
    return max(abs(v1 - v2) for v1, v2 in zip(V1, V2))

def print_grid_values(V, grid_size, title):
    print(title)
    for r in range(grid_size[0]):
        row_values = [f"{V[r * grid_size[1] + c]:.2f}" for c in range(grid_size[1])]
        print(" | ".join(row_values))

def print_grid_policy(policy, grid_size, action_names, title):
    print(title)
    for r in range(grid_size[0]):
        row_policy = [action_names[policy[r * grid_size[1] + c]] for c in range(grid_size[1])]
        print(" | ".join(row_policy))