"""Localhost-only workbench console and control API."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from .core.controller import LabController
ROOT=Path(__file__).resolve().parents[1]; CONTROLLER=LabController(ROOT)
try: CONTROLLER.probe_local()
except Exception as exc: CONTROLLER.events.append("WORKBENCH_PROBED",{"status":"FAILED","error":repr(exc)},workbench_id="local_rtx5060")
HTML='''<!doctype html><html><head><meta charset="utf-8"><title>GPU Workbench</title><style>body{font:15px system-ui;margin:0;background:#101318;color:#eef}.bar{padding:18px;background:#182233;position:sticky;top:0}main{padding:18px;max-width:1100px;margin:auto}.card{background:#1c2430;border:1px solid #39485a;border-radius:8px;padding:14px;margin:12px 0}button{padding:8px;margin:4px}pre{white-space:pre-wrap}</style></head><body><div class="bar"><b>LOCAL GPU OPERATOR WORKBENCH</b><pre id="banner">loading</pre></div><main><div class="card"><h2>Dashboard</h2><button onclick="refresh()">Refresh</button><button onclick="harvest()">Harvest</button><pre id="state"></pre></div><div class="card"><h2>Human directive</h2><input id="msg" style="width:70%"><button onclick="directive()">Send Directive</button></div><div class="card"><h2>Recent events</h2><pre id="events"></pre></div></main><script>async function get(p){return (await fetch(p)).json()}async function refresh(){let x=await get('/api/state');document.querySelector('#state').textContent=JSON.stringify(x,null,2);document.querySelector('#banner').textContent=JSON.stringify(x.workbench||{status:'UNVERIFIED ENVIRONMENT'},null,2);document.querySelector('#events').textContent=JSON.stringify((await get('/api/events')).slice(-20),null,2)}async function directive(){await fetch('/api/agent/directive',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:document.querySelector('#msg').value})});refresh()}async function harvest(){await fetch('/api/harvest',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({campaign_id:'rms_norm_train__rtx5060_sm120__cuda_cpp'})});refresh()}refresh();setInterval(refresh,2000)</script></body></html>'''
def body(req):
    n=int(req.headers.get('Content-Length','0')); return json.loads(req.rfile.read(n) or b'{}')
class Handler(BaseHTTPRequestHandler):
    def send_json(self,x,code=200):
        data=json.dumps(x,ensure_ascii=False).encode(); self.send_response(code); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(data)
    def do_GET(self):
        p=urlparse(self.path).path
        if p=='/': data=HTML.encode(); self.send_response(200); self.send_header('Content-Type','text/html; charset=utf-8'); self.end_headers(); self.wfile.write(data)
        elif p in ('/api/state','/api/status'): self.send_json(CONTROLLER.status())
        elif p=='/api/events': self.send_json(CONTROLLER.events.replay())
        elif p=='/api/knowledge': self.send_json({"source":"lab/index.json","status":"READY"})
        elif p=='/api/context': self.send_json({"status":"READY","source":"context_builder"})
        elif p=='/api/campaigns': self.send_json(["rms_norm_train__rtx5060_sm120__cuda_cpp"])
        else: self.send_json({"error":"not found"},404)
    def do_POST(self):
        p=urlparse(self.path).path; data=body(self)
        if p=='/api/agent/directive': self.send_json(CONTROLLER.directive(data.get('text','')))
        elif p=='/api/agent/ask': self.send_json(CONTROLLER.ask_record(data.get('text','')))
        elif p=='/api/harvest': self.send_json({"path":CONTROLLER.harvest(data.get('campaign_id','rms_norm_train__rtx5060_sm120__cuda_cpp'))})
        elif p=='/api/pause': self.send_json(CONTROLLER.events.append('PAUSE_REQUESTED',data,workbench_id='local_rtx5060'))
        elif p=='/api/stop': self.send_json(CONTROLLER.events.append('STOP_REQUESTED',data,workbench_id='local_rtx5060'))
        elif p=='/api/emergency-stop': self.send_json(CONTROLLER.events.append('STOP_REQUESTED',{'emergency':True,**data},workbench_id='local_rtx5060'))
        else: self.send_json({"error":"not found"},404)
    def log_message(self,*args): pass
def main(host='127.0.0.1',port=8765): print(f'GPU workbench UI: http://{host}:{port}/'); ThreadingHTTPServer((host,port),Handler).serve_forever()
if __name__=='__main__': main()
