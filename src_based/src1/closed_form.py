import numpy as np

def solve_closed_form(policy, cell_types, gamma=0.9, boundary_reward=-1.0, goal_reward=1.0, forbidden_reward=-1.0):
    """
    Evaluate a deterministic policy by closed-form solution v = (I - gamma P)^-1 r.

    policy: (25,) array of actions 'U','D','L','R' for each cell (ignored for goal/forbidden cells)
    cell_types: (25,) array with values 'N' (normal), 'G' (goal), 'F' (forbidden). Word forms are also accepted.
    returns v: (25,) state-value vector
    """
    def _norm(ct):
        # accept either single-letter ('N','G','F') or words ('normal','goal','forbidden')
        if ct in ('N', 'G', 'F'):
            return ct
        mapping = {'normal': 'N', 'goal': 'G', 'forbidden': 'F'}
        return mapping.get(ct, 'N')
    n = 25
    P = np.zeros((n, n), dtype=float)
    r = np.zeros(n, dtype=float)

    def rc_to_s(r0, c0):
        return r0 * 5 + c0

    # helper to compute next state and immediate reward
    for s in range(n):
        r0 = s // 5
        c0 = s % 5
        ctype = _norm(cell_types[s])
        # Only goals ('G') are absorbing terminals. Forbidden ('F') are non-terminal but entering them gives a penalty.
        if ctype == 'G':
            P[s, s] = 1.0
            r[s] = goal_reward
            continue

        a = policy[s]
        # determine next cell
        nr, nc = r0, c0
        if a == 'U':
            nr -= 1
        elif a == 'D':
            nr += 1
        elif a == 'L':
            nc -= 1
        elif a == 'R':
            nc += 1
        else:
            # if unknown, stay
            pass

        # check boundary
        if nr < 0 or nr >= 5 or nc < 0 or nc >= 5:
            # hitting boundary: remain in same state and receive boundary reward
            P[s, s] = 1.0
            r[s] = boundary_reward
        else:
            s2 = rc_to_s(nr, nc)
            P[s, s2] = 1.0
            # reward is defined as reward of the arrived cell
            if _norm(cell_types[s2]) == 'G':
                r[s] = goal_reward
            elif _norm(cell_types[s2]) == 'F':
                r[s] = forbidden_reward
            else:
                r[s] = 0.0

    I = np.eye(n)
    A = I - gamma * P
    # solve A v = r
    v = np.linalg.solve(A, r)
    return v


if __name__ == '__main__':
    # quick demo default policy (all right)
    policy = np.array(['R'] * 25)
    cell_types = np.array(['N'] * 25)
    # set center as goal
    cell_types[12] = 'G'
    v = solve_closed_form(policy, cell_types)
    print(v.reshape((5,5)))
