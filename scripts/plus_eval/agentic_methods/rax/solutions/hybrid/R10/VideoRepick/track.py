import numpy as np
from percep import frame_points, occupancy, blobs, X0, Y0, CELL

def occ_points(occ):
    iy,ix=np.nonzero(occ>0)
    return np.stack([X0+(ix+0.5)*CELL, Y0+(iy+0.5)*CELL],1)

def grip_sum(fr): return fr['state'][6]+fr['state'][7]

def find_pick_release(frames, close_thr=0.076):
    """Return (grasp_idx, release_idx, release_xy) from per-frame state in video."""
    vid=[i for i,f in enumerate(frames) if f['phase']=='video_demo']
    g=[grip_sum(frames[i]) for i in vid]
    closed=[i for i,s in zip(vid,g) if s<close_thr]
    if not closed: return None
    gi=closed[0]
    rel=None
    for i in vid:
        if i>gi and grip_sum(frames[i])>=close_thr:
            rel=i; break
    if rel is None: return None
    return gi, rel, np.array(frames[gi]['state'][:2]), np.array(frames[rel]['state'][:2])

def track(frames, log=None, gate=0.045):
    pr=find_pick_release(frames)
    if pr is None: raise RuntimeError('no pick found')
    gi,rel,grasp_xy,rel_xy=pr
    # baseline count from first frame
    b0,_=blobs(occupancy(frame_points(frames[0])))
    N=len(b0)
    # start: first frame after release where blob count == N, robot settled
    start=None
    for i in range(rel, len(frames)):
        b,_=blobs(occupancy(frame_points(frames[i])))
        if len(b)==N and i>rel+2:
            start=i; break
    if start is None: start=rel+3; b,_=blobs(occupancy(frame_points(frames[start])))
    pos=np.array([[x,y] for x,y,c in b]); vel=np.zeros_like(pos)
    tgt=int(np.argmin(np.linalg.norm(pos-rel_xy,axis=1)))
    hist=[(start,pos.copy())]
    prev_t=frames[start].get('source_frame_index',start)
    for i in range(start+1,len(frames)):
        fr=frames[i]
        if fr['phase']!='video_demo': break
        t=fr.get('source_frame_index',i); dt=max(1,t-prev_t); prev_t=t
        pts=occ_points(occupancy(frame_points(fr)))
        pred=pos+vel*dt
        c=pred.copy(); got=np.zeros(len(c),bool)
        for it in range(4):
            if len(pts)==0: break
            D=np.linalg.norm(pts[:,None,:]-c[None,:,:],axis=2)
            a=D.argmin(1); dm=D.min(1)
            for k in range(len(c)):
                sel=(a==k)&(dm<gate)
                if sel.sum()>=4:
                    c[k]=pts[sel].mean(0); got[k]=True
        newv=(c-pos)/dt
        vel=np.where(got[:,None],0.5*vel+0.5*newv,vel*0.8)
        pos=c
        hist.append((i,pos.copy()))
    return dict(target=tgt,N=N,start=start,grasp=gi,release=rel,grasp_xy=grasp_xy.tolist(),
                release_xy=rel_xy.tolist(),final=pos.tolist(),target_xy=pos[tgt].tolist(),hist=hist)
