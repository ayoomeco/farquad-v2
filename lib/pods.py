"""list | terminate <id...> | terminate-all-run  : RunPod pods (no keys printed)."""
import sys, os, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from home import gj_home
sys.path.insert(0, gj_home() + '/engine')
from app.providers.runpod_pod import RunPodClient
c = RunPodClient()
def ls():
    d = c.gql('query { myself { pods { id name desiredStatus costPerHr machine { gpuDisplayName } } } }')
    d = d.get('data', d) if isinstance(d, dict) else d
    return d['myself']['pods']
if __name__ == '__main__' and sys.argv[1] == 'list':
    P = ls(); print('balance', c.balance()); [print(p['id'], p['name'], p['desiredStatus'], p.get('costPerHr'), (p.get('machine') or {}).get('gpuDisplayName')) for p in P]; print('count', len(P))
elif __name__ == '__main__' and sys.argv[1] == 'terminate':
    for i in sys.argv[2:]: print(i, c.terminate_pod(i))
