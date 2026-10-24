"""Viral 5b V1: GJ-Fast linked (same as Viral 5: Singularity + [realism .7 live-action] + swap v1.5 + pro4500 + turbo .8, Sage, 6 steps, 1152x640),
NO face mesh, NO Motion Repair, ONE character pass per clip (lead + optional 2nd), linked 141-f pieces (step 119, 22-f AddGuide), NATURAL MOUTH (muted driver, no audio inputs)."""
import json, copy, pathlib, os
FH=os.environ.get('FARQUAD_HOME',os.path.expanduser('~/farquad'))
G=pathlib.Path(FH+'/plan/graphs')
BASE=json.load(open(FH+'/plan/base_graphs/gb_S1_t2.api.json'))
A=json.load(open(FH+'/plan/base_graphs/gjB_p1_s42.api.json'))
LEN,STEP=141,119
CLIPS={
 'lalaland':dict(look="Photoreal live-action, lit like the scene.",subj=[("the man in the white shirt and tie","Gomez Addams","gomez"),("the woman in the yellow dress","Morticia Addams","morticia")],
   keep="The hilltop, dusk sky, city lights, lighting and camera stay as in <Video 1>."),
 'pulp':dict(look="Live-action costumed performers with stop-motion style character design, lit like the scene.",subj=[("the man in the black suit dancing on the right","Jack Skellington","jack"),("the woman in the white shirt and black trousers dancing on the left","Sally","sally")],
   keep="The diner stage, audience, lighting and camera stay as in <Video 1>."),
 'thisisamerica':dict(look="Photoreal cinematic CGI character, lit like the scene.",subj=[("the shirtless dancing man in light trousers","Thanos, wearing exactly the clothes shown in <Picture 1>","thanos")],
   keep="The warehouse stays exactly as in <Video 1>: add no people, keep it as empty or as full as <Video 1> in every frame. Same framing, lighting and camera as <Video 1>."),
 'walken':dict(look="Photoreal live-action, lit like the scene.",subj=[("the dancing man in the suit","Patrick Bateman","bateman")],
   keep="The hotel lobby, lighting and camera stay as in <Video 1>."),
 'singleladies':dict(look="Photoreal live-action, black-and-white like the scene.",subj=[("the dancing woman in the center","Michael Myers","myers")],
   keep="The two dancers on the left and right stay exactly as they are. The white studio, lighting and camera stay as in <Video 1>.")}
REAL={'lalaland':True,'walken':True,'singleladies':True,'thisisamerica':False,'pulp':False}
FIX={'orig':" Replace each Subject completely, head, face, hair and hands.",'bg':" Keep the exact original scene and colours of <Video 1> in every frame.",
     'leak':" Never show the reference picture itself; only the scene of <Video 1>."}
def prompt(slug):
    c=CLIPS[slug]; s=[]; refs=[]
    for j,(w,n,key) in enumerate(c['subj'],1):
        refs.append(f'v5b_{key}.png'); s.append(f"Subject {j} is {w}; Subject {j} becomes {n} in <Picture {j}>.")
    return ("In <Video 1>: "+" ".join(s)+" Each Subject keeps the motion, gestures, head turns, expressions and natural mouth movement of <Video 1>. "+c['keep']+" "+c['look']), refs
def graph(slug,k,guide,seed=42,fixes=(),t=0):
    g=copy.deepcopy(BASE); i=g['15']['inputs']; pr,refs=prompt(slug); pr+=''.join(FIX[f] for f in fixes)
    i.update(prompt=pr,length=LEN,width=1152,height=640)
    for x in [x for x in list(i) if x.startswith('ref_images.')]: del i[x]
    for nd in ('5','50','52'): g.pop(nd,None)
    for j,im in enumerate(refs):
        n=str(200+j); g[n]={'class_type':'LoadImage','inputs':{'image':im}}; i[f'ref_images.ref_image_{j}']=[n,0]
    chain=([('h3-realism-people-t2v-i2v-r2v.safetensors',0.7)] if REAL[slug] else [])+[('h3_character_swap_v1.5.safetensors',1.0),('h3_character_swap_pro4500_1000.safetensors',1.0),('minimax_h3_turbo_v4_step600_ema.safetensors',0.8)]
    for x in ('90','91','92','93'): g.pop(x,None)
    prev=['1',0]
    for j,(ln,s) in enumerate(chain):
        n=str(90+j); g[n]={'class_type':'LoraLoaderModelOnly','inputs':{'model':prev,'lora_name':ln,'strength_model':s}}; prev=[n,0]
    g['54']={'class_type':'PathchSageAttentionKJ','inputs':{'model':prev,'sage_attention':'auto'}}
    TG=os.environ.get("RIDTAG",""); d=f"v5b_{slug}{TG}_d{k}.mp4"; g['6']['inputs']['file']=d; g['22']['inputs']['noise_seed']=seed; g['21']['inputs']['steps']=6
    m=['54',0]; g['21']['inputs']['model']=m; g['23']['inputs']['model']=m; g['23']['inputs']['conditioning']=['15',0]
    up=[d]+refs
    if guide:
        g['70']={'class_type':'LoadVideo','inputs':{'file':guide}}; g['71']={'class_type':'GetVideoComponents','inputs':{'video':['70',0]}}
        g['72']=copy.deepcopy(A['72']); g['23']['inputs']['conditioning']=['72',0]; up.append(guide)
    rid=f"v5b_{slug}{TG}_p{k}_t{t}"; g['33']['inputs']['filename_prefix']=f'v5b/{rid}'
    p=G/f'{rid}.api.json'; p.write_text(json.dumps(g))
    return {'id':rid,'engine':'l21','seed':seed,'graph_path':str(p),'version':rid,'uploads':up,
            'params':{'steps':6,'length':LEN,'W':1152,'H':640},'est_s':60+3*LEN,'protect':True,'rung':rid,'tier':'L',
            'needs_stage2':False,'question':f"v5b {slug} p{k} t{t} {list(fixes)}",'spec':{}}
if __name__=='__main__':
    for s in CLIPS: print(s,'|',prompt(s)[0],'\n')
