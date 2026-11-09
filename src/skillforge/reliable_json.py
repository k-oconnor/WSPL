"""Bounded atomic JSON writes; optional status telemetry never aborts a run."""
import json
import os
from pathlib import Path
import tempfile
import time


def save(path,value):
    path=Path(path)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    temp=Path(name)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        for attempt in range(8):
            try:os.replace(temp,path);return
            except OSError as error:
                if not isinstance(error,PermissionError) and getattr(error,'winerror',None) not in (5,32,33):raise
                if attempt==7:raise
                time.sleep(min(.05*2**attempt,.5))
    finally:
        try:temp.unlink(missing_ok=True)
        except OSError:pass


def status(path,value):
    try:save(path,value);return True
    except OSError as error:
        # Status is telemetry. Essential artifacts still use fail-closed save().
        value['status_write_failures']=value.get('status_write_failures',0)+1
        value['last_status_write_error']=str(error)
        return False
