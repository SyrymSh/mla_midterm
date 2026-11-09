"""Execute the notebook in a fresh local kernel and compare a repeated full fit.

Run after pip install -r requirements.txt. Uses this interpreter for the kernel.
No credentials or network requests are needed. Local Jupyter sockets are required.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs'
notebook=ROOT/'notebooks/Video_Game_Endterm.ipynb'
before=pd.read_csv(OUT/'model_comparison.csv').set_index('model')
pred_before=pd.read_csv(OUT/'test_predictions.csv')
with tempfile.TemporaryDirectory(prefix='endterm-kernel-') as temporary:
    work=Path(temporary)
    kernel=work/'kernels'/'endterm-check'
    kernel.mkdir(parents=True)
    (kernel/'kernel.json').write_text(json.dumps({
        'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
        'display_name':'Endterm check','language':'python',
        'env':{'PYTHONPATH':os.environ.get('PYTHONPATH',''),'PYTHONDONTWRITEBYTECODE':'1',
               'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}}),encoding='utf-8')
    os.environ['JUPYTER_PATH']=str(work)
    os.environ['JUPYTER_RUNTIME_DIR']=str(work/'runtime')
    os.environ['IPYTHONDIR']=str(work/'ipython')
    nb=nbformat.read(notebook,as_version=4)
    def progress(cell,cell_index):
        if cell.cell_type=='code': print('Executing notebook cell',cell_index+1,flush=True)
    client=NotebookClient(nb,timeout=2400,startup_timeout=90,kernel_name='endterm-check',
        resources={'metadata':{'path':str(ROOT)}},on_cell_start=progress)
    client.execute()
    nb.metadata['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
    nbformat.write(nb,notebook)
code=[c for c in nb.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code)
assert not [o for c in code for o in c.outputs if o.output_type=='error']
after=pd.read_csv(OUT/'model_comparison.csv').set_index('model')
columns=['train_cv_MAE_hours','train_cv_MAE_std_hours','test_MAE_hours','test_RMSE_hours','test_R2','test_MedAE_hours']
delta=np.abs(before[columns].sort_index()-after[columns].sort_index()).to_numpy().max()
np.testing.assert_allclose(before[columns].sort_index(),after[columns].sort_index(),rtol=1e-9,atol=1e-9)
pred_after=pd.read_csv(OUT/'test_predictions.csv')
assert pred_before.game_id.tolist()==pred_after.game_id.tolist()
pred_delta=float(np.abs(pred_before.predicted_hours-pred_after.predicted_hours).max())
np.testing.assert_allclose(pred_before.predicted_hours,pred_after.predicted_hours,rtol=1e-9,atol=1e-9)
sys.path.insert(0,str(ROOT/'src'))
from common import sha256,utc_now,write_json
write_json(OUT/'reproducibility_check.json',{
    'verified_at_utc':utc_now(),'method':'Two complete fits: command-line then fresh-kernel Restart/Run All with RETRAIN=True.',
    'code_cells_executed':len(code),'errors':0,'models_compared':len(after),
    'max_numeric_metric_difference':float(delta),'max_selected_prediction_difference':pred_delta,
    'dataset_sha256':sha256(ROOT/'data/clean/games_enriched.csv'),
    'split_sha256':sha256(ROOT/'data/clean/fixed_split.csv'),
    'source_sha256':{p.name:sha256(p) for p in sorted((ROOT/'src').glob('*.py'))},
    'notebook_sha256':sha256(notebook)})
print('Full notebook execution and repeated-fit comparison passed:',len(code),'code cells.',flush=True)
