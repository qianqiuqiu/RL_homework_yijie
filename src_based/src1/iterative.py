import numpy as np

def solve_iterative(policy, cell_types, gamma=0.9, tol=1e-6, max_iter=10000, boundary_reward=-1.0, goal_reward=1.0, forbidden_reward=-1.0):
    """
    Policy evaluation by iterative updates: v_{k+1} = r + gamma * P * v_k
    Returns v (25,) and number of iterations performed.
    """
    def _norm(ct):
        if ct in ('N', 'G', 'F'):
            return ct
        mapping = {'normal': 'N', 'goal': 'G', 'forbidden': 'F'}
        return mapping.get(ct, 'N')
    n = 25
    P = np.zeros((n, n), dtype=float)
    r = np.zeros(n, dtype=float)

    def rc_to_s(r0, c0):
        return r0 * 5 + c0

    for s in range(n):
        r0 = s // 5
        c0 = s % 5
        ctype = _norm(cell_types[s])
        if ctype == 'G':
            P[s, s] = 1.0
            r[s] = goal_reward
            continue

        a = policy[s]
        nr, nc = r0, c0
        if a == 'U':
            nr -= 1
        elif a == 'D':
            nr += 1
        elif a == 'L':
            nc -= 1
        elif a == 'R':
            nc += 1

        if nr < 0 or nr >= 5 or nc < 0 or nc >= 5:
            P[s, s] = 1.0
            r[s] = boundary_reward
        else:
            s2 = rc_to_s(nr, nc)
            P[s, s2] = 1.0
            if _norm(cell_types[s2]) == 'G':
                r[s] = goal_reward
            elif _norm(cell_types[s2]) == 'F':
                r[s] = forbidden_reward
            else:
                r[s] = 0.0

    v = np.zeros(n, dtype=float)
    for it in range(1, max_iter+1):
        v_new = r + gamma * (P @ v)
        delta = np.max(np.abs(v_new - v))
        v = v_new
        if delta < tol:
            return v, it
    return v, max_iter


if __name__ == '__main__':
    import numpy as _np
    policy = _np.array(['R'] * 25)
    cell_types = _np.array(['N'] * 25)
    cell_types[12] = 'G'
    v, it = solve_iterative(policy, cell_types)
    print('iters', it)
    print(v.reshape((5,5)))
