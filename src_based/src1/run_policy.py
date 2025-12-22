import numpy as np
from PIL import Image, ImageDraw, ImageFont

# grid size
N = 5

    # cell_types: 5x5, 每个元素为:
    # 'N' = normal, 'G' = goal (entering得到 +1, treated as terminal),
    # 'F' = forbidden (entering得到 -1, NOT treated as terminal here — it still has actions and a value)
    # 示例：把(1,2)设为 goal, (2,1)设为 forbidden（索引从0开始）
cell_types = [
    ['N','N','N','N','N'],
    ['N','F','F','N','N'],
    ['N','N','F','N','N'],
    ['N','F','G','F','N'],
    ['N','F','N','N','N'],
]

# actions: 5x5，非终端格子的动作（'U','D','L','R'）
# 对于 G 和 F 单元，action 会被忽略（视为终端）
# 示例：默认右移，中心位置设为 'U'
actions = [
    ['R','R','R','D','D'],
    ['U','U','R','D','D'],
    ['U','L','D','R','D'],
    ['U','R','R','L','D'],
    ['U','R','U','L','L'],
]

# 参数
gamma = 0.9
boundary_reward = -1
goal_reward = +1
forbidden_reward = -1

# 辅助映射
act2delta = {'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}

def build_P_r(cell_types, actions):
    n = N*N
    P = np.zeros((n,n))
    r = np.zeros(n)
    for i in range(N):
        for j in range(N):
            s = i*N + j
            typ = cell_types[i][j]
            # Only goals are treated as absorbing terminals here. Forbidden cells ('F') are NOT terminal —
            # they have actions and state-values like normal cells, but entering them yields forbidden_reward.
            if typ == 'G':
                # treat goal as absorbing terminal (value determined by entering reward)
                P[s,s] = 1.0
                r[s] = 0.0
            else:
                a = actions[i][j] if actions[i][j] in act2delta else 'R'
                di,dj = act2delta[a]
                ni, nj = i+di, j+dj
                if not (0 <= ni < N and 0 <= nj < N):
                    # hit boundary: stay and get boundary_reward
                    P[s,s] = 1.0
                    r[s] = boundary_reward
                else:
                    ns = ni*N + nj
                    P[s,ns] = 1.0
                    # reward depends on the cell you move INTO
                    if cell_types[ni][nj] == 'G':
                        r[s] = goal_reward
                    elif cell_types[ni][nj] == 'F':
                        r[s] = forbidden_reward
                    else:
                        r[s] = 0.0
    return P, r

def closed_form_eval(P, r, gamma):
    n = len(r)
    A = np.eye(n) - gamma * P
    v = np.linalg.solve(A, r)
    return v

def iterative_eval(P, r, gamma, tol=1e-6, max_iter=10000):
    n = len(r)
    v = np.zeros(n)
    for it in range(max_iter):
        v_next = r + gamma * (P @ v)
        if np.max(np.abs(v_next - v)) < tol:
            return v_next, it+1
        v = v_next
    return v, max_iter

def draw_values_and_policy(v, cell_types, actions, out_path='policy_values.jpg'):
    cell_w = 120
    img_w = N * cell_w
    img_h = N * cell_w
    img = Image.new('RGB', (img_w, img_h), 'white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    # draw grid, fill goal/forbidden
    for i in range(N):
        for j in range(N):
            x0 = j*cell_w; y0 = i*cell_w
            x1 = x0 + cell_w; y1 = y0 + cell_w
            typ = cell_types[i][j]
            if typ == 'G':
                draw.rectangle([x0,y0,x1,y1], fill=(173,235,173))
            elif typ == 'F':
                draw.rectangle([x0,y0,x1,y1], fill=(250,180,180))
            else:
                draw.rectangle([x0,y0,x1,y1], fill=(255,255,255))
            draw.rectangle([x0,y0,x1,y1], outline=(0,0,0), width=2)

            # draw value text
            s = v[i*N + j]
            txt = f"{s:.3f}"
            try:
                bbox = draw.textbbox((0,0), txt, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except Exception:
                tw, th = font.getsize(txt)
            draw.text((x0 + 6, y0 + 6), txt, fill=(0,0,0), font=font)

    # draw policy arrows for non-terminals
    for i in range(N):
        for j in range(N):
            typ = cell_types[i][j]
            if typ in ('G','F'):
                continue
            a = actions[i][j] if actions[i][j] in act2delta else 'R'
            di,dj = act2delta[a]
            # arrow from center toward direction
            cx = j*cell_w + cell_w/2
            cy = i*cell_w + cell_w/2
            ex = cx + dj * cell_w * 0.28
            ey = cy + di * cell_w * 0.28
            draw.line([(cx,cy),(ex,ey)], fill=(20,100,20), width=6)
            # triangle head
            # small perp vector
            px = -di
            py = dj
            head_len = 12
            p1 = (ex, ey)
            p2 = (ex - dj*head_len + px*head_len/1.8, ey - di*head_len + py*head_len/1.8)
            p3 = (ex - dj*head_len - px*head_len/1.8, ey - di*head_len - py*head_len/1.8)
            draw.polygon([p1,p2,p3], fill=(20,100,20))

    img.save(out_path, quality=95)
    print("Saved", out_path)

def main():
    P, r = build_P_r(cell_types, actions)
    # closed-form
    v_cf = closed_form_eval(P, r, gamma)
    draw_values_and_policy(v_cf.reshape(-1), cell_types, actions, 'policy_values_closed.jpg')
    print("Closed form done.")

    # iterative
    v_it, iters = iterative_eval(P, r, gamma)
    draw_values_and_policy(v_it.reshape(-1), cell_types, actions, 'policy_values_iter.jpg')
    print("Iterative done. iters =", iters)

if __name__ == "__main__":
    main()