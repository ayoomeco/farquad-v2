import json,time,pathlib,sys,os
R=pathlib.Path(sys.argv[1]); G=str(next((pathlib.Path(os.environ.get('FARQUAD_HOME',os.path.expanduser('~/farquad')))/'plan'/'graphs').glob('*.api.json')))
while not (R/'KEEPALIVE_OFF').exists() and not (R/'STOP').exists():
    rid=f'keepalive_{int(time.time())}'
    with open(R/'SKIP.txt','a') as f: f.write(rid+'\n')
    with open(R/'extra_runs.jsonl','a') as f: f.write(json.dumps({'id':rid,'engine':'l21','graph_path':G,'uploads':[],'est_s':1})+'\n')
    time.sleep(90)
