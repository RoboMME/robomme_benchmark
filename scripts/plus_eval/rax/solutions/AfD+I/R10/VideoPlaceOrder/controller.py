import sys, os, json, time, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0, str(HERE))
import numpy as np
from robot import parse, deproject, deproject_mask
from motion import Robot
from perception import blobs, cube_mask
from plan import parse_instruction, load_sequence, solve

OUT=Path('/workspace/output'); 
def mklog():
    try: OUT.mkdir(parents=True, exist_ok=True); f=open(OUT/'controller_log.txt','a')
    except Exception: f=None
    def log(*a):
        s=' '.join(str(x) for x in a); print(s, flush=True)
        if f: f.write(s+'\n'); f.flush()
    return log

def cube_world(meta, imgs, color):
    a=imgs[('front','rgb')]; dep=imgs[('front','depth')]
    m=cube_mask(a,color)
    from scipy import ndimage
    lab,n=ndimage.label(m)
    if n==0: return None
    sz=ndimage.sum(m,lab,np.arange(1,n+1)); j=int(np.argmax(sz))+1
    pw=deproject_mask(meta,dep,'front',lab==j)
    if pw is None or len(pw)<5: return None
    zt=np.percentile(pw[:,2],90)
    top=pw[pw[:,2]>zt-0.008]
    return np.array([np.median(top[:,0]),np.median(top[:,1]),zt])

def main(ctx):
    log=mklog(); t0=time.time()
    r=Robot(ctx,log); meta=r.observe()
    ins=meta.get('instruction',''); color,n=parse_instruction(ins)
    log('instruction:',ins,'->',color,n)
    if color is None: color='red'
    homes=[]; ans=None
    try:
        mp=(meta.get('frame_sequence') or {}).get('manifest_path')
        if mp is None:
            import glob
            c=sorted(glob.glob('/workspace/runtime/autonomous_controller/runs/*/frames/sequence_000000/manifest.json'), key=os.path.getmtime)
            mp=os.path.relpath(c[-1],'/workspace'); log('fallback manifest',mp)
        F,T,ex,FM=load_sequence(mp,log=log)
        if ex is None: ex=r.imgs[('front','rgb')]
        log('video frames',len(F))
        homes,ans,info=solve(F,T,ex,color,n,log=log,FM=FM)
        log('answer home',ans, homes[ans] if ans is not None else None, 'time %.1f'%(time.time()-t0))
    except Exception as e:
        log('PLAN ERROR',repr(e)); log(traceback.format_exc())
    a=r.imgs[('front','rgb')]; dep=r.imgs[('front','depth')]
    if not homes:
        homes=[(x,y) for x,y,s in blobs(a) if s>=50]
    if ans is None:
        ans=0 if homes else None
    if ans is None:
        log('no target found'); return
    tw=deproject(meta,dep,'front',homes[ans][0],homes[ans][1],r=1)
    log('target world',np.round(tw,4))
    cw=cube_world(meta,r.imgs,color)
    log('cube world',None if cw is None else np.round(cw,4))
    if cw is None: return
    pick_place(r, cw, tw, color, log)
    log('done complete=%s ended=%s steps_left=%s time %.1f'%(r.complete,r.ended,r.remaining(),time.time()-t0))

def pick_place(r, cw, tw, color, log, attempts=3):
    for att in range(attempts):
        if r.ended: return
        gz=max(0.015, cw[2]-0.022)
        r.move_to([cw[0],cw[1],0.15],-1)
        r.move_to([cw[0],cw[1],gz+0.05],-1)
        r.move_to([cw[0],cw[1],gz],-1, max_step=0.015)
        r.grip(1,10)
        r.move_to([cw[0],cw[1],0.15],1)
        if r.ended: return
        w=r.state[6]+r.state[7]
        log(f'attempt {att} grip width {w:.4f}')
        if w<0.01:
            log('grasp failed; reacquire'); r.grip(-1,6)
            r.move_to([cw[0],cw[1],0.25],-1)
            r.observe(); c2=cube_world(r.meta,r.imgs,color)
            if c2 is not None: cw=c2
            continue
        r.move_to([tw[0],tw[1],0.15],1)
        r.move_to([tw[0],tw[1],0.06],1, max_step=0.015)
        r.grip(-1,10)
        if r.ended: return
        r.move_to([tw[0],tw[1],0.18],-1)
        if r.ended: return
        r.step([[0,0,0,0,0,0,-1]]*3)
        if r.ended: return
        r.observe(); c2=cube_world(r.meta,r.imgs,color)
        log('after place: cube', None if c2 is None else np.round(c2,3), 'complete', r.complete)
        if c2 is None: return
        if np.linalg.norm(c2[:2]-tw[:2])<0.02:
            r.step([[0,0,0,0,0,0,-1]]*5); return
        cw=c2
