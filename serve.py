"""Local chat and training stats; model loading is lazy and CPU-only."""
import argparse,json,threading,time
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'out';RUNS=ROOT/'runs';SMOKE=False
CHAT_LOCK=threading.Lock();CHAT_MODEL=None;CHAT_MTIME=None

def read_json(path,default):
    try:return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):return default

class Handler(BaseHTTPRequestHandler):
    def send(self,code,body,kind='application/json'):
        if isinstance(body,dict):body=json.dumps(body,allow_nan=False).encode()
        self.send_response(code);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(body)
    def do_GET(self):
        path=self.path.split('?')[0]
        if path in ('/','/index.html','/stats','/stats.html'):
            name='stats.html' if 'stats' in path else 'index.html'
            content=(ROOT/'public'/name).read_text(encoding='utf-8')
            if SMOKE:content=content.replace('AgmGPT-5-It','AgmGPT-5-It (smoke test)')
            return self.send(200,content.encode(),'text/html; charset=utf-8')
        if path=='/api/status':return self.send(200,dict(model='AgmGPT-5-It',smoke=SMOKE,available=(OUT/'agmgpt_5_it.pt').exists(),backend='nanochat',weights='floating_point'))
        if path=='/api/stats':
            state=RUNS/'training.json';info=read_json(state,{})
            rows=[]
            try:
                with (RUNS/'metrics.jsonl').open(encoding='utf-8') as f:
                    for line in f:
                        try:rows.append(json.loads(line))
                        except ValueError:pass
            except OSError:pass
            live=rows[-1] if rows else {};vals=[{'step':r['step'],'val_loss':r['val_loss']} for r in rows if 'val_loss' in r]
            age=max(0,time.time()-(RUNS/'metrics.jsonl').stat().st_mtime) if rows else 0
            return self.send(200,dict(available=bool(info),info=info,live=live,validation=vals[-500:],training_history=rows[-1000:],update_age_seconds=age))
        self.send(404,dict(error='Not found'))
    def do_POST(self):
        global CHAT_MODEL,CHAT_MTIME
        if self.path!='/api/chat':return self.send(404,dict(error='Not found'))
        # Same-origin requests only; no public tunnel is configured automatically.
        origin=self.headers.get('Origin');host=self.headers.get('Host')
        if origin and origin not in ('http://'+host,'https://'+host):return self.send(403,dict(error='Invalid origin'))
        if not CHAT_LOCK.acquire(blocking=False):return self.send(409,dict(error='A reply is already generating'))
        started=False
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=32000:raise ValueError('Invalid request size')
            data=json.loads(self.rfile.read(size));text=data.get('message');variant=data.get('model','instruct')
            if variant!='instruct':raise ValueError('Only AgmGPT-5-It is available')
            if not isinstance(text,str) or not text.strip() or len(text)>4000:raise ValueError('Enter a message of 1–4000 characters')
            path=OUT/'agmgpt_5_it.pt'
            if not path.exists():return self.send(503,dict(error='No floating-point checkpoint yet. Train the upgraded model first. Legacy binary checkpoints are incompatible.'))
            history=data.get('history',[])
            if not isinstance(history,list):raise ValueError('Invalid history')
            history=[h for h in history[-3:] if isinstance(h,dict) and isinstance(h.get('user'),str) and isinstance(h.get('ai'),str)]
            import torch
            from tokenizers import Tokenizer
            from checkpoint import load_model,tokenizer_hash
            self.send_response(200);self.send_header('Content-Type','application/x-ndjson; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Connection','close');self.end_headers();self.close_connection=True;started=True
            def event(value):self.wfile.write((json.dumps(value)+'\n').encode());self.wfile.flush()
            event(dict(type='loading'))
            mtime=path.stat().st_mtime_ns
            if CHAT_MODEL is None or CHAT_MTIME!=mtime:CHAT_MODEL=load_model(path,expected_hash=tokenizer_hash(ROOT/'data/tokenizer.json'));CHAT_MTIME=mtime
            tok=Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
            context=''.join(f'<|user|> {h["user"]} <|ai|> {h["ai"]} <|eos|>\n\n' for h in history)
            ids=tok.encode(context+f'<|user|> {text} <|ai|>').ids
            x=torch.tensor([ids[-CHAT_MODEL.config.block_size:]],dtype=torch.long)
            stops={tok.token_to_id(s) for s in ('<|eos|>','<|user|>')};tokens=[]
            for token in CHAT_MODEL.generate_stream(x,stop_ids=stops):
                tokens.append(token);reply=tok.decode(tokens)
                if not reply.endswith('\ufffd'):event(dict(type='token',text=reply))
            event(dict(type='done',reply=tok.decode(tokens).strip(),checkpoint='agmgpt_5_it.pt'))
        except (BrokenPipeError,ConnectionResetError):pass
        except Exception as error:
            if started:
                try:event(dict(type='error',error=str(error)))
                except (BrokenPipeError,ConnectionResetError):pass
            else:self.send(400,dict(error=str(error)))
        finally:CHAT_LOCK.release()

def start_server(port=8785,smoke=False,out_dir=None,runs_dir=None):
    global OUT,RUNS,SMOKE,CHAT_MODEL,CHAT_MTIME
    SMOKE=smoke;OUT=Path(out_dir) if out_dir else ROOT/('out-smoke' if smoke else 'out');RUNS=Path(runs_dir) if runs_dir else ROOT/('runs-smoke' if smoke else 'runs')
    CHAT_MODEL=None;CHAT_MTIME=None
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    actual=server.server_address[1]
    print(f'AgmGPT-5-It chat: http://127.0.0.1:{actual}/',flush=True)
    print(f'Training stats: http://127.0.0.1:{actual}/stats',flush=True)
    return server

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8785);p.add_argument('--smoke',action='store_true');args=p.parse_args();server=start_server(args.port,smoke=args.smoke)
    try:threading.Event().wait()
    except KeyboardInterrupt:server.shutdown()
