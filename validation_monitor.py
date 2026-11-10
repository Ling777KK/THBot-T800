"""Optional fixed validation and plateau stopping for the original PPO trainer."""
import json
import math
from pathlib import Path
import torch
from memory_eval import FixedEvaluator


class ValidationMonitor:
    def __init__(self,args,env,out):
        if args.arch!='cnn' or args.bomb or args.dodger or env.vec_dim!=3 or env.n_actions!=18:
            raise ValueError('Fixed validation mode currently supports move-only CNN experiments')
        self.args=args;self.out=Path(out)
        self.interval=max(1,math.ceil(args.validate_every/(args.rollout*args.envs)))
        target=dict(vars(args))
        if args.validation_lagf:target['lagf']=args.validation_lagf
        if args.validation_timing_profile:
            target['timing_profile']=args.validation_timing_profile
            target['lagf']=''
        if getattr(args,'validation_timing_trace',''):
            target['timing_trace']=args.validation_timing_trace
        meta=dict(stack=env.stack,lags=env.lags,args=target)
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.evaluator=FixedEvaluator(meta,args.validation_seed,args.validation_per_condition,args.validation_frames)
        if getattr(args,'future',0):
            import future_env
            future_env.attach(self.evaluator.env,args.validation_seed,fixed=True)
        self.history=[];self.best=float('-inf');self.best_steps=0
        self.anchor=float('-inf');self.anchor_steps=0
        self.minimum=int(args.plateau_min*1e6);self.patience=int(args.plateau*1e6)
        self.train_seconds=0.
        (self.out/'checkpoints').mkdir(exist_ok=True)

    def check(self,net,steps,update,pack,force=False):
        if not force and update%self.interval:return False
        r=self.evaluator.evaluate(net)
        score=r['mean_s']
        if not math.isfinite(score):raise ValueError('Validation produced a nonfinite score')
        improved=score>self.best
        if improved:self.best=score;self.best_steps=steps
        delta=max(.2,.01*abs(self.anchor)) if math.isfinite(self.anchor) else 0.
        if score>self.anchor+delta:
            self.anchor=score;self.anchor_steps=max(steps,self.minimum)
        stop=self.patience>0 and steps>=self.minimum and steps-self.anchor_steps>=self.patience
        r.update(steps=steps,update=update,best_score=self.best,best_steps=self.best_steps,
                 plateau_anchor=self.anchor,plateau_anchor_steps=self.anchor_steps,
                 train_seconds=self.train_seconds,stop=stop)
        self.history.append(r)
        (self.out/'validation.json').write_text(json.dumps(self.history,indent=2),encoding='utf-8')
        light=pack(False)
        light.update(validation_best=self.best,validation_best_steps=self.best_steps)
        torch.save(light,self.out/'checkpoints'/f'step_{steps:09d}.pt')
        full=pack(True)
        full.update(validation_best=self.best,validation_best_steps=self.best_steps)
        torch.save(full,self.out/'last.pt')
        if improved:torch.save(full,self.out/'best.pt')
        print(f"VALIDATION steps={steps/1e6:.3f}M score={score:.3f}s best={self.best:.3f}s@{self.best_steps/1e6:.3f}M "
              f"cap={r['cap_pct']:.1f}% eval={r['seconds']:.1f}s stop={stop}",flush=True)
        return stop
