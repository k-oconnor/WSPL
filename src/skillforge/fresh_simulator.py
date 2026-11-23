"""Trusted local simulator RPC with graceful Windows venv process shutdown."""
import json
import queue
import subprocess
import threading


class Simulator:
    def __init__(self, config, out, manifest, identity):
        self.log = (out/'simulator.stderr.log').open('wb')
        self.events = (out/'simulator_rpc.jsonl').open('x', encoding='utf-8')
        try:
            self.process = subprocess.Popen(
                [config['sim_python'], '-I', config['sim_worker'],
                 '--manifest', str(manifest), '--sha256', identity],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, encoding='utf-8', bufsize=1,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except BaseException:
            self.log.close()
            self.events.close()
            raise
        self.queue = queue.Queue()
        def read():
            try:
                for line in self.process.stdout:
                    self.queue.put(line)
            finally:
                self.queue.put(None)
        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()

    def call(self, request):
        self.events.write(json.dumps(dict(request=request))+'\n')
        self.events.flush()
        self.process.stdin.write(json.dumps(request)+'\n')
        self.process.stdin.flush()
        try:
            line = self.queue.get(timeout=60)
        except queue.Empty:
            raise RuntimeError('Physics worker timeout; no retry') from None
        if line is None:
            raise RuntimeError('Physics worker exited')
        reply = json.loads(line)
        self.events.write(json.dumps(dict(reply=reply))+'\n')
        self.events.flush()
        if not reply.get('ok'):
            raise RuntimeError(reply.get('error'))
        return reply['result']

    def close(self):
        try:
            if self.process.poll() is None:
                try:
                    # Do not terminate only the venv redirector and orphan its worker.
                    self.process.stdin.write('{"op":"close"}\n')
                    self.process.stdin.flush()
                    self.process.stdin.close()
                    self.process.wait(timeout=15)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    if self.process.poll() is None:
                        subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                        self.process.wait(timeout=15)
            self.reader.join(timeout=15)
            if self.reader.is_alive():
                raise RuntimeError('Simulator stdout reader did not exit')
        finally:
            for stream in (self.process.stdin, self.process.stdout, self.log, self.events):
                stream.close()
