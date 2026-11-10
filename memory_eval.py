"""Fixed, disjoint-seed validation/test runner for CNN and GRU policies.

Evaluation never auto-resets dead episodes. Thus deaths in one lane cannot
consume reset RNG and change the emitters in another still-alive lane.
"""
import argparse
import json
import time
from pathlib import Path
import torch
from danmaku_env import DanmakuEnv, TRAIN_V11, OOD_V2, HELDOUT_FAMILIES, FAMILY_NAMES
from memory_policy import build_policy,step_policy


def environment_kwargs(source):
    a=source.get('args',source)
    def numbers(key,default,kind=float):
        v=a.get(key,default)
        return [kind(x) for x in v.split(',')] if isinstance(v,str) and v else (v or None)
    return dict(stack=source.get('stack',7),lags=source.get('lags',[30,16,8,4,2,1,0]),
                blind=numbers('blind','5.3,5.3,5.3'),big_draw=numbers('big_draw','48,160',int),
                hit_scale=a.get('hit_scale',.8),ymin_play=a.get('ymin_play',16.),
                lat_p=a.get('lat_p',0.),lat2_frac=a.get('lat2_frac',0.),
                lagf=numbers('lagf',''),fs_jitter=a.get('fs_jitter',0.),
                art_p=0.,shot_p=0.,bomb_mode=False,timing_profile=a.get('timing_profile') or None,
                timing_trace=a.get('timing_trace') or None)


class FixedEvaluator:
    def __init__(self,source,seed=7001,per_condition=4,cap_frames=2400,diffs=(.9,1.2,1.4),include_ood=False):
        self.seed=int(seed)
        self.cap_frames=int(cap_frames)
        self.conditions=[(int(f),float(d),r) for f in sorted(set(TRAIN_V11+(OOD_V2+HELDOUT_FAMILIES if include_ood else [])))
                         for d in diffs for r in range(per_condition)]
        self.env=DanmakuEnv(n_envs=len(self.conditions),max_frames=cap_frames,frame_skip=int(source.get('args',{}).get('frame_skip',2)),
                            d_max=max(diffs),**environment_kwargs(source))
        self.env.set_conditions([[f] for f,d,r in self.conditions],[d for f,d,r in self.conditions])

    @torch.no_grad()
    def evaluate(self,net,reset_memory_every_step=False,keep_samples=False):
        was_training=net.training
        net.eval()
        t0=time.perf_counter()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(self.seed)
            obs=self.env.reset()
            original_reset=self.env._reset
            self.env._reset=lambda mask: None
            n=self.env.N
            first_done=torch.zeros(n,device=self.env.dev,dtype=torch.bool)
            length=torch.full((n,),float(self.cap_frames),device=self.env.dev)
            hit=torch.zeros(n,device=self.env.dev,dtype=torch.bool)
            state=None
            starts=torch.ones(n,device=self.env.dev,dtype=torch.bool)
            try:
                limit=self.cap_frames*4+1 if self.env.timing_clock is not None else self.cap_frames//self.env.fs+1
                for k in range(limit):
                    if reset_memory_every_step:
                        state=None
                    logits,_,state=step_policy(net,obs,state,starts)
                    obs,_,done,info=self.env.step(logits.argmax(-1))
                    new=done & ~first_done
                    length=torch.where(new,info['ep_frames'].float().clamp(max=self.cap_frames),length)
                    hit |= new & info['hit']
                    first_done |= done
                    starts=done
                    if k % 32 == 31 and bool(first_done.all()):
                        break
            finally:
                self.env._reset=original_reset
        net.train(was_training)
        seconds=(length/60).cpu().tolist()
        hit_cpu=hit.cpu().tolist()
        groups={}
        for key in sorted({(f,d) for f,d,r in self.conditions}):
            ids=[i for i,c in enumerate(self.conditions) if c[:2]==key]
            groups[f'{FAMILY_NAMES[key[0]]}@{key[1]}']=dict(mean_s=sum(seconds[i] for i in ids)/len(ids),
                                                        cap_pct=100*sum(not hit_cpu[i] for i in ids)/len(ids))
        result=dict(mean_s=sum(seconds)/len(seconds),
                    cap_pct=100*sum(not v for v in hit_cpu)/len(hit_cpu),
                    seed=self.seed,episodes=n,cap_frames=self.cap_frames,
                    seconds=time.perf_counter()-t0,groups=groups)
        for label,families in (('train',TRAIN_V11),('heldout',OOD_V2+HELDOUT_FAMILIES),('late',list(range(29,35)))):
            ids=[i for i,c in enumerate(self.conditions) if c[0] in families]
            if ids:
                result[label+'_mean_s']=sum(seconds[i] for i in ids)/len(ids)
        if keep_samples:
            result['samples']=[dict(family=f,difficulty=d,replicate=r,survival_s=s,hit=h)
                               for (f,d,r),s,h in zip(self.conditions,seconds,hit_cpu)]
        return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=9001)
    p.add_argument('--per_condition',type=int,default=16)
    p.add_argument('--cap_frames',type=int,default=7200)
    p.add_argument('--include_ood',action='store_true')
    p.add_argument('--reset_memory',action='store_true')
    a=p.parse_args()
    torch.set_num_threads(4)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    net=build_policy(ck).eval()
    evaluator=FixedEvaluator(ck,a.seed,a.per_condition,a.cap_frames,include_ood=a.include_ood)
    result=evaluator.evaluate(net,a.reset_memory,keep_samples=True)
    result['checkpoint']=str(Path(a.checkpoint).resolve())
    result['reset_memory']=a.reset_memory
    Path(a.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('groups','samples')},indent=2),flush=True)


if __name__=='__main__':
    main()
