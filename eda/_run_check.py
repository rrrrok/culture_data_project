import json, sys, traceback
import matplotlib
matplotlib.use('Agg')

nb = json.load(open('eda.ipynb', encoding='utf-8'))
code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']

g = {'__name__': '__main__'}
for i, c in enumerate(code_cells):
    src = ''.join(c['source'])
    try:
        exec(compile(src, f'<cell {i}>', 'exec'), g)
        print(f'[OK] cell {i}')
    except Exception:
        print(f'[FAIL] cell {i}')
        traceback.print_exc()
        sys.exit(1)
print('ALL CELLS OK')
