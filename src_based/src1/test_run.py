import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import closed_form
import iterative

CELL_SIZE = 80

def draw_values(v, types, policy, outpath):
    img = Image.new('RGB', (CELL_SIZE*5, CELL_SIZE*5), 'white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 18)
    except Exception:
        font = ImageFont.load_default()

    for r in range(5):
        for c in range(5):
            x0 = c*CELL_SIZE
            y0 = r*CELL_SIZE
            x1 = x0 + CELL_SIZE
            y1 = y0 + CELL_SIZE
            idx = r*5 + c
            if types[idx] == 'G':
                fill = (173,216,230)
            elif types[idx] == 'F':
                fill = (255,200,150)
            else:
                fill = (255,255,255)
            draw.rectangle([x0,y0,x1,y1], fill=fill, outline='black')
            txt = f'{v[idx]:.3f}'
            try:
                bbox = draw.textbbox((0,0), txt, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            except Exception:
                w, h = font.getsize(txt)
            draw.text((x0 + (CELL_SIZE-w)/2, y0 + (CELL_SIZE-h)/2), txt, fill='black', font=font)
            # draw policy arrow for non-terminal cells
            a = policy[idx]
            if types[idx] in ('N','F') and a in ('U','D','L','R'):
                cx = x0 + CELL_SIZE/2
                cy = y0 + CELL_SIZE/2
                L = CELL_SIZE * 0.28
                if a == 'U':
                    ex, ey = cx, cy - L
                elif a == 'D':
                    ex, ey = cx, cy + L
                elif a == 'L':
                    ex, ey = cx - L, cy
                else:
                    ex, ey = cx + L, cy
                draw.line((cx, cy, ex, ey), fill='black', width=3)
                # arrowhead
                ah = 6
                if a == 'U' or a == 'D':
                    sign = -1 if a == 'U' else 1
                    p1 = (ex - ah, ey - sign*ah)
                    p2 = (ex + ah, ey - sign*ah)
                else:
                    sign = -1 if a == 'L' else 1
                    p1 = (ex - sign*ah, ey - ah)
                    p2 = (ex - sign*ah, ey + ah)
                draw.polygon([ (ex,ey), p1, p2 ], fill='black')

    img.save(outpath, quality=95)


def main():
    # demo policy: arrows pointing right everywhere
    policy = np.array(['R'] * 25)
    types = np.array(['N'] * 25)
    # set center as goal and some forbidden cells around
    types[12] = 'G'
    types[6] = 'F'
    types[18] = 'F'

    print('Running closed-form solver...')
    v_closed = closed_form.solve_closed_form(policy, types, gamma=0.9)
    out1 = os.path.join(os.getcwd(), 'policy_values_closed.jpg')
    draw_values(v_closed, types, policy, out1)
    print('Saved', out1)

    print('Running iterative solver...')
    v_iter, its = iterative.solve_iterative(policy, types, gamma=0.9)
    out2 = os.path.join(os.getcwd(), 'policy_values_iter.jpg')
    draw_values(v_iter, types, policy, out2)
    print('Saved', out2, 'iterations', its)


if __name__ == '__main__':
    main()
