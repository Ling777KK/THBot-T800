"""Joint sensor-age / actuator-delay profiles, measured in 60Hz game frames."""
import json
import math
from pathlib import Path


def load_profile(profile):
    if isinstance(profile,(str,Path)):
        obj=json.loads(Path(profile).read_text(encoding='utf-8-sig'))
        rows=obj['rows'] if isinstance(obj,dict) else obj
    else:rows=profile
    result=[]
    for row in rows:
        if isinstance(row,dict):
            s=row['sensor_frames'];a=row['action_frames'];w=row.get('weight',row.get('count',0))
        else:s,a,w=row
        if int(s)!=s or int(a)!=a or not 0<=s<=8 or not 0<=a<=8 or not math.isfinite(w) or w<0:
            raise ValueError('Invalid measured timing-profile row')
        result.append((int(s),int(a),float(w)))
    if not result or sum(r[2] for r in result)<=0:raise ValueError('Timing profile needs positive total weight')
    return result
