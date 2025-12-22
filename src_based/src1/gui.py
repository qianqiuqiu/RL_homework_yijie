import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import closed_form
import iterative
import os

CELL_SIZE = 80

def default_policy_and_types():
    policy = np.array(['R'] * 25)
    types = np.array(['N'] * 25)
    # sample: center is goal
    types[12] = 'G'
    return policy, types

class GridGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('5x5 Policy Evaluation')
        self.policy = np.array([''] * 25)
        self.types = np.array(['N'] * 25)
        self.gamma = tk.DoubleVar(value=0.9)
        self.create_widgets()

    def create_widgets(self):
        frm = ttk.Frame(self)
        frm.pack(padx=10, pady=10)

        self.cells = []
        for r in range(5):
            row = []
            for c in range(5):
                f = ttk.Frame(frm, width=CELL_SIZE, height=CELL_SIZE, relief='ridge')
                f.grid(row=r, column=c, padx=1, pady=1)
                # entry for type
                etype = tk.StringVar(value='N')
                ein = ttk.Combobox(f, values=['N','G','F'], width=9, textvariable=etype)
                ein.pack(side='top')
                # entry for action
                action = tk.StringVar(value='R')
                aentry = ttk.Combobox(f, values=['U','D','L','R',''], width=9, textvariable=action)
                aentry.pack(side='top')
                row.append((etype, action))
            self.cells.append(row)

        ctrl = ttk.Frame(self)
        ctrl.pack(padx=10, pady=6)

        ttk.Label(ctrl, text='gamma:').grid(row=0, column=0)
        ttk.Entry(ctrl, textvariable=self.gamma, width=6).grid(row=0, column=1)

        self.solver = tk.StringVar(value='closed')
        ttk.Radiobutton(ctrl, text='Closed-form', variable=self.solver, value='closed').grid(row=0, column=2)
        ttk.Radiobutton(ctrl, text='Iterative', variable=self.solver, value='iter').grid(row=0, column=3)

        run_btn = ttk.Button(ctrl, text='Run and Save JPG', command=self.on_run)
        run_btn.grid(row=0, column=4, padx=6)

    def read_inputs(self):
        policy = np.array([''] * 25)
        types = np.array(['N'] * 25)
        for r in range(5):
            for c in range(5):
                idx = r*5 + c
                etype, action = self.cells[r][c]
                types[idx] = etype.get()
                policy[idx] = action.get() or ' '
        return policy, types

    def on_run(self):
        policy, types = self.read_inputs()
        gamma = float(self.gamma.get())
        # sanitize
        for i in range(25):
            if types[i] not in ('N','G','F'):
                messagebox.showerror('Input error', f'Cell {i} has invalid type')
                return
            # only clear action for goal (terminal). forbidden cells keep their action and will show arrows/values
            if types[i] == 'G':
                policy[i] = ' '
            elif policy[i].strip() == '':
                policy[i] = 'R'

        if self.solver.get() == 'closed':
            v = closed_form.solve_closed_form(policy, types, gamma=gamma)
        else:
            v, it = iterative.solve_iterative(policy, types, gamma=gamma)

        # draw image
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
                # color goal/forbidden
                if types[idx] == 'G':
                    fill = (173,216,230)
                elif types[idx] == 'F':
                    fill = (255,200,150)
                else:
                    fill = (255,255,255)
                draw.rectangle([x0,y0,x1,y1], fill=fill, outline='black')
                val = v[idx]
                txt = f'{val:.3f}'
                try:
                    bbox = draw.textbbox((0,0), txt, font=font)
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                except Exception:
                    w, h = font.getsize(txt)
                draw.text((x0 + (CELL_SIZE-w)/2, y0 + (CELL_SIZE-h)/2), txt, fill='black', font=font)
                # draw policy arrow for normal and forbidden cells (forbidden are non-terminal now)
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

        out = os.path.join(os.getcwd(), 'policy_values.jpg')
        img.save(out, quality=95)
        messagebox.showinfo('Saved', f'Saved image to {out}')

if __name__ == '__main__':
    app = GridGUI()
    app.mainloop()
