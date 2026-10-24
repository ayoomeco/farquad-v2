import http.server,os,subprocess,threading,urllib.parse
P=os.environ.get('PSK','x');W='/w';os.makedirs(W+'/logs',exist_ok=True);N=[0]
class H(http.server.BaseHTTPRequestHandler):
    def ok(self):
        if self.headers.get('X-P')!=P: self.send_response(403);self.end_headers();return False
        return True
    def do_PUT(self):
        if not self.ok():return
        p=W+'/'+urllib.parse.unquote(self.path[3:]);os.makedirs(os.path.dirname(p),exist_ok=True)
        n=int(self.headers['Content-Length']);f=open(p,'wb')
        while n>0:
            b=self.rfile.read(min(n,1<<20));f.write(b);n-=len(b)
        f.close();self.send_response(200);self.end_headers();self.wfile.write(b'ok')
    def do_POST(self):
        if not self.ok():return
        c=self.rfile.read(int(self.headers['Content-Length'])).decode();N[0]+=1;L=f'{W}/logs/{N[0]}.log'
        subprocess.Popen(['bash','-c',c+f'; echo EXIT=$? >> {L}'],stdout=open(L,'w'),stderr=subprocess.STDOUT,cwd=W)
        self.send_response(200);self.end_headers();self.wfile.write(str(N[0]).encode())
    def do_GET(self):
        if not self.ok():return
        p=W+'/'+urllib.parse.unquote(self.path[3:])
        if not os.path.isfile(p): self.send_response(404);self.end_headers();return
        self.send_response(200);self.send_header('Content-Length',str(os.path.getsize(p)));self.end_headers()
        with open(p,'rb') as f:
            while b:=f.read(1<<20): self.wfile.write(b)
http.server.ThreadingHTTPServer(('0.0.0.0',8000),H).serve_forever()
