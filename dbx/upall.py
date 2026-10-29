"""Farquad uploader: Dropbox temporary-upload-link + one curl per file (no chunked connector uploads).

Token mode (generic):
  DROPBOX_TOKEN=<token from env> python3 upall.py <DEST_DIR> <file> [<file> ...]
    -> for each file: get a temporary upload link for <DEST_DIR>/<file name>, POST the bytes with one curl,
       then create (or reuse) a share link. Prints one JSON line per file: {"file","path","http","share"}.
    DEST_DIR is a Dropbox folder such as /Farquad/<job>. The token is read from the environment only
    and is never printed or written to disk. Temporary upload links take files up to 150 MB.

Links mode (the original V1 loop, for when a connector hands out the temporary links):
  python3 upall.py --links
    Run in a folder holding all_items.json ([{"i","src","off","len"}]) and all_links.json ({"<i>": "<link id>"}).
    Uploads every item (bytes off..off+len of src) to https://content.dropboxapi.com/apitul/1/<link id>,
    up to 2 rounds per item, 4 in parallel, results in all_results.json; exits after ~20 idle minutes."""
import json,subprocess,time,os,sys,urllib.request,urllib.error
from concurrent.futures import ThreadPoolExecutor

def links_mode():
    it={str(x['i']):x for x in json.load(open('all_items.json'))}
    R=json.load(open('all_results.json')) if os.path.exists('all_results.json') else {}
    def up(k,link):
        x=it[k]
        for a in range(5):
            c=subprocess.run(f"tail -c +{x['off']+1} \"$S\" | head -c {x['len']} | curl -s -o /dev/null -w '%{{http_code}}' -X POST -H 'Content-Type: application/octet-stream' --data-binary @- 'https://content.dropboxapi.com/apitul/1/{link}'",shell=True,capture_output=True,text=True,env={**os.environ,'S':x['src']}).stdout
            if c=='200': break
            time.sleep(15 if c=='429' else 3)
        time.sleep(0.3); return k,c
    idle=0
    while idle<120:
        L=json.load(open('all_links.json'))
        todo=[(k,v) for k,v in L.items() if R.get(k)!='200' and R.get(k+'_tries',0)<2]
        if not todo: idle+=1; time.sleep(10); continue
        idle=0
        with ThreadPoolExecutor(4) as ex:
            for k,c in ex.map(lambda a:up(*a),todo):
                R[k]=c; R[k+'_tries']=R.get(k+'_tries',0)+1
        json.dump(R,open('all_results.json','w'))
        print(time.strftime('%T'),'ok',sum(1 for k,v in R.items() if not k.endswith('_tries') and v=='200'),flush=True)

def api(endpoint,body):
    tok=os.environ.get('DROPBOX_TOKEN','').strip()
    if not tok: sys.exit('DROPBOX_TOKEN is not set in the environment')
    req=urllib.request.Request('https://api.dropboxapi.com/2/'+endpoint,data=json.dumps(body).encode(),method='POST',
                               headers={'Authorization':'Bearer '+tok,'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=60) as r: return r.status,json.loads(r.read() or b'{}')
    except urllib.error.HTTPError as e:
        try: return e.code,json.loads(e.read() or b'{}')
        except Exception: return e.code,{}

def share(path):
    st,d=api('sharing/create_shared_link_with_settings',{'path':path})
    if st==200: return d.get('url')
    st,d=api('sharing/list_shared_links',{'path':path,'direct_only':True})
    L=d.get('links') or []
    return L[0]['url'] if st==200 and L else None

def token_mode(dest,files):
    dest='/'+dest.strip('/')
    for f in files:
        path=f"{dest}/{os.path.basename(f)}"; c='skip'
        if os.path.getsize(f)>150*1024*1024:
            print(json.dumps({'file':f,'path':path,'http':'too_big_for_temp_link'}),flush=True); continue
        for a in range(5):
            st,d=api('files/get_temporary_upload_link',{'commit_info':{'path':path,'mode':'overwrite','autorename':False,'mute':True},'duration':3600})
            if st!=200 or 'link' not in d: c=f'link_{st}'; time.sleep(15 if st==429 else 3); continue
            c=subprocess.run(['curl','-s','-o','/dev/null','-w','%{http_code}','-X','POST','-H','Content-Type: application/octet-stream',
                              '--data-binary','@'+f,d['link']],capture_output=True,text=True).stdout
            if c=='200': break
            time.sleep(15 if c=='429' else 3)
        print(json.dumps({'file':f,'path':path,'http':c,'share':share(path) if c=='200' else None}),flush=True)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--links': links_mode()
    elif len(sys.argv)>2: token_mode(sys.argv[1],sys.argv[2:])
    else: sys.exit(__doc__)
