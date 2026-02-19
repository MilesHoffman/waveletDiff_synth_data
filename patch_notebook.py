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
                if 'NOISE_SCHEDULE = "exponential"' in line:
                    new_source.append('NOISE_PRIOR = "student-t" # @param ["gaussian", "student-t"]\n')
                    new_source.append('NU = 3.0 # @param {type:"number"}\n')
                    new_source.append('LOSS_TYPE = "huber" # @param ["mse", "huber", "logcosh"]\n')
                    new_source.append('HUBER_DELTA = 1.0 # @param {type:"number"}\n')
            cell['source'] = new_source

        is_run_cell = any('Cell 3: Run WaveletDiff Training' in line for line in source)
        if is_run_cell:
            new_source = []
            for line in source:
                if '--noise_schedule {NOISE_SCHEDULE} \\' in line.replace(' ', ''):
                    new_source.append(line)
                    new_source.append('    --noise_prior {NOISE_PRIOR} \\\n')
                    new_source.append('    --nu {NU} \\\n')
                    new_source.append('    --loss_type {LOSS_TYPE} \\\n')
                    new_source.append('    --huber_delta {HUBER_DELTA} \\\n')
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print('Notebook successfully updated.')
