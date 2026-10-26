import json,time,pathlib,sys,os
sys.path.insert(0, str(next(p/'lib' for p in pathlib.Path(__file__).resolve().parents if (p/'lib'/'home.py').is_file())))
from home import gj_home
R=pathlib.Path(sys.argv[1]); G=str(next((pathlib.Path(gj_home())/'plan'/'graphs').glob('*.api.json')))
while not (R/'KEEPALIVE_OFF').exists() and not (R/'STOP').exists():
    rid=f'keepalive_{int(time.time())}'
    with open(R/'SKIP.txt','a') as f: f.write(rid+'\n')
    with open(R/'extra_runs.jsonl','a') as f: f.write(json.dumps({'id':rid,'engine':'l21','graph_path':G,'uploads':[],'est_s':1})+'\n')
    time.sleep(90)
