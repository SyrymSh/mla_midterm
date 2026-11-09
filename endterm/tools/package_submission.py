"""Create both Moodle archives with identical complete contents and verify bytes.

Run after full model/notebook/material validation. Archives are written beside
the project folder. Dependencies, build caches and credentials are excluded.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.python_packages','.venv','__pycache__','.git','.ipynb_checkpoints','node_modules','.build','.runtime','tmp'}

def files():
    result=[]
    for current,dirs,names in os.walk(ROOT):
        dirs[:]=sorted(d for d in dirs if d not in EXCLUDED and not d.startswith('.chart-data-'))
        for d in dirs:
            path=Path(current)/d
            assert not path.is_symlink() and not path.is_junction(),path
        for name in sorted(names):
            if name.endswith(('.pyc','.log','.tmp')) or name.startswith(('~$','.~lock.')):continue
            if name.startswith('.env') and name!='.env.example':continue
            path=Path(current)/name
            assert not path.is_symlink(),path
            assert path.resolve().is_relative_to(ROOT.resolve()),path
            result.append(path)
    return result

def digest(data):return hashlib.sha256(data).hexdigest()

inventory=ROOT/'outputs/submission_inventory.json'
entries=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())}
         for p in files() if p!=inventory]
inventory.write_text(json.dumps({'scope':'All included files except this inventory itself.',
    'files':entries},indent=2),encoding='utf-8')
allfiles=files()
paths={p.relative_to(ROOT).as_posix() for p in allfiles}
required={'src/collect.py','reference/midterm_source/src/collect_igdb.py',
    'reference/Endterm Rubric.pdf','data/clean/games_model_inputs.csv','data/clean/fixed_split.csv',
    'outputs/search/random_forest.csv','outputs/search/gradient_boosting.csv',
    'outputs/cluster_error_summary.csv','outputs/all_model_error_examples.csv',
    'outputs/reproducibility_check.json','docs/CHANGELOG.md','docs/CONTRIBUTIONS.md',
    'docs/Endterm_Report.pdf','docs/Endterm_Defense.pptx','docs/Endterm_Defense.pdf',
    'notebooks/Video_Game_Endterm.ipynb','tools/presentation/build_deck.mjs'}
assert required.issubset(paths),required-paths
first=ROOT.parent/'Endterm_Shadiyarbek_Syrym_SE2424.zip'
second=ROOT.parent/'Endterm_Akhanov_Yerdaulet_SE2424.zip'
with zipfile.ZipFile(first,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for p in allfiles:archive.write(p,'Video_Game_Endterm/'+p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(first) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist())==len(allfiles)
    for p in allfiles:
        assert archive.read('Video_Game_Endterm/'+p.relative_to(ROOT).as_posix())==p.read_bytes(),p
    validation=json.loads(archive.read('Video_Game_Endterm/outputs/validation_summary.json'))
    for p,sha in validation['deliverable_sha256'].items():
        assert digest(archive.read('Video_Game_Endterm/'+p))==sha,p
shutil.copyfile(first,second)
assert digest(first.read_bytes())==digest(second.read_bytes())
print(json.dumps({'archives':[str(first),str(second)],'files':len(allfiles),
    'MB':round(first.stat().st_size/1048576,2),'identical_contents_and_bytes':True,
    'sha256':digest(first.read_bytes())},indent=2))
