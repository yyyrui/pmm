"""Execute the notebook's definition/result-display cells without retraining."""
import os
from pathlib import Path
os.environ['JUPYTER_PATH']=str(Path('.jupyter').absolute())
os.environ['IPYTHONDIR']=str(Path('.ipython').absolute())
os.environ['JUPYTER_RUNTIME_DIR']=str(Path('.jupyter/runtime').absolute())
import nbformat
from nbclient import NotebookClient
path=Path('ＰＰＰ.ipynb')
notebook=nbformat.read(path,as_version=4)
client=NotebookClient(notebook,timeout=45,kernel_name='pmm-experiment',
    resources={'metadata':{'path':str(Path('.').absolute())}})
client.execute()
nbformat.write(notebook,path)
print('Notebook definitions and recorded-results display executed successfully; no retraining.')
