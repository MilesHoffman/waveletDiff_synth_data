import json

path = r'c:\Users\kotan\personal\repos\waveletDiff_synth_data\waveletDiff_training.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        is_config_cell = any('Cell 1: Global Configuration' in line for line in source)
        if is_config_cell:
            new_source = []
            for line in source:
                new_source.append(line)
                if 'USE_CROSS_LEVEL_ATTENTION = True' in line:
                    new_source.append('EXPLORATION_RATIO = 0.3 # @param {type:"number"}\n')
                    new_source.append('ADAPTIVE_START_PCT = 0.8 # @param {type:"number"}\n')
            cell['source'] = new_source

        is_run_cell = any('Cell 3: Run WaveletDiff Training' in line for line in source)
        if is_run_cell:
            new_source = []
            for line in source:
                if '--energy_weight {ENERGY_WEIGHT} \\' in line.replace(' ', ''):
                    new_source.append(line)
                    new_source.append('    --exploration_ratio {EXPLORATION_RATIO} \\\n')
                    new_source.append('    --adaptive_start_pct {ADAPTIVE_START_PCT} \\\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print('Notebook successfully updated with Adaptive params.')
