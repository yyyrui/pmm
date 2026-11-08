"""Execute the improved notebook in-process, without a network kernel server."""
# Configure all runtime/cache directories before importing IPython or nbformat.
from pmm_robustness import ROOT, OUT, RUNTIME
import os
import json
import time
os.environ['JUPYTER_RUNTIME_DIR'] = str(RUNTIME / 'jupyter-runtime')
os.environ['JUPYTER_DATA_DIR'] = str(RUNTIME / 'jupyter-data')
os.environ['JUPYTER_PATH'] = str(RUNTIME / 'jupyter-data')
import nbformat
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

os.chdir(ROOT)
path = ROOT / 'PMM_改善版.ipynb'
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
shell = InteractiveShell.instance()
count = 0
for cell in notebook.cells:
    if cell.cell_type != 'code':
        continue
    count += 1
    started = time.time()
    with capture_output(stdout=True, stderr=True, display=True) as captured:
        result = shell.run_cell(cell.source, store_history=False)
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(f'Notebook cell {count} failed: {captured.stderr} {captured.stdout}')
    outputs = []
    if captured.stdout:
        outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
    if captured.stderr:
        outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
    for output in captured.outputs:
        outputs.append(nbformat.v4.new_output('display_data', data=output.data,
                                             metadata=output.metadata))
    cell.outputs = outputs
    cell.execution_count = count
    print(f'Executed cell {count}: {time.time() - started:.2f}s; {len(outputs)} outputs', flush=True)
chosen = shell.user_ns['chosen']
expected = pd.read_parquet(OUT / 'test' /
    f"{chosen['name']}_{chosen['policy']}_s1_predictions.parquet")
pd.testing.assert_frame_equal(shell.user_ns['prediction'], expected, check_exact=True)
nbformat.validate(notebook)
nbformat.write(notebook, path)
(OUT / 'notebook_execution.json').write_text(json.dumps(dict(
    notebook=str(path.relative_to(ROOT)), code_cells=count, all_passed=True,
    execution='IPython in-process; no listening socket or external notebook server',
    training_started=False, predictions_match_evaluated_model_exactly=True),
    ensure_ascii=False, indent=2))
print('EXECUTED', path, flush=True)
