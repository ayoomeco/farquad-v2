"""Write every `File: <path>` block from Farquad script skills to disk.
Usage: python3 extract.py <dest dir> <skill.md> [<skill.md> ...]
A block is the line `File: <relative path>` (optionally followed by ` (chunk i/n)` and/or ` (no trailing newline)`)
and then a fenced code block. Chunks of one file are joined in order. Prints each path and byte count."""
import re,sys,os
HDR=re.compile(r'^File: (\S+)((?: \([^)]*\))*)\s*$')
def parse(text):
    out={}; L=text.split('\n'); i=0
    while i<len(L):
        m=HDR.match(L[i])
        if not m: i+=1; continue
        path,flags=m.group(1),m.group(2); i+=1
        while i<len(L) and not L[i].strip(): i+=1
        fm=re.match(r'^(`{3,})',L[i]); fence=fm.group(1); i+=1; body=[]
        while L[i]!=fence: body.append(L[i]); i+=1
        i+=1; s='\n'.join(body)+'\n'
        if 'no trailing newline' in flags: s=s[:-1]
        out[path]=out.get(path,'')+s
    return out
if __name__=='__main__':
    dest=sys.argv[1]; files={}
    for md in sys.argv[2:]:
        for p,s in parse(open(md,encoding='utf-8').read()).items(): files[p]=files.get(p,'')+s
    for p,s in files.items():
        assert not os.path.isabs(p) and '..' not in p.split('/'), p
        fp=os.path.join(dest,p); os.makedirs(os.path.dirname(fp) or '.',exist_ok=True)
        open(fp,'w',encoding='utf-8',newline='').write(s)
        if p.endswith('.sh'): os.chmod(fp,0o755)
        print(p,len(s.encode()))
