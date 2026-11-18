SINGLE_DT=99
SINGLE_MIN=100
ADAPT=True
import numpy as np
from perception import target_mask

def prep(H, shape=(256,256), home_r=7.0):
    yy,xx=np.mgrid[0:shape[0],0:shape[1]]
    near=np.zeros(shape,bool)
    for h in H: near|=(xx-h[0])**2+(yy-h[1])**2<home_r**2
    disks=[((xx-h[0])**2+(yy-h[1])**2<5.0**2) for h in H]
    return near,disks

def frame_scores(a, H, near, disks, big_px=0, rad_k=0):
    m=target_mask(a)
    N=len(H)
    fill=[m[d].mean() for d in disks]
    off=m&~near
    if big_px:
        from scipy import ndimage
        lab,nl=ndimage.label(off)
        if nl:
            sz=ndimage.sum(off,lab,np.arange(1,nl+1))
            bad=np.isin(lab,1+np.nonzero(sz>big_px)[0])
            off=off&~bad
    ys,xs=np.nonzero(off)
    sc={}
    if len(xs)<30: return sc, fill
    S=set(zip(xs.tolist(),ys.tolist()))
    for i in range(N):
        for j in range(i+1,N):
            if fill[i]>0.35 or fill[j]>0.35: continue
            M=(H[i]+H[j]); sep=np.linalg.norm(H[i]-H[j])
            rx=np.round(M[0]-xs).astype(int); ry=np.round(M[1]-ys).astype(int)
            # pixel at least 0.3*sep away from midpoint
            r2=((xs-M[0]/2)**2+(ys-M[1]/2)**2)
            far=r2>(0.3*sep)**2
            if rad_k: far&=r2<(rad_k*sep)**2
            hit=sum(1 for x,y,f in zip(rx.tolist(),ry.tolist(),far.tolist()) if f and (x,y) in S)
            if hit>=30: sc[(i,j)]=hit
    return sc, fill

def detect_swaps2(frames, times, homes, gap=1.0, max_span=2.0, min_votes=2, min_span=0.8, big_px=0, rad_k=0, blocked=None):
    H=np.array(homes,float); near,disks=prep(H)
    if ADAPT and len(times)>2:
        dt=float(np.median(np.diff(times)))
        min_span=min(min_span, max(0.25, 1.15-2*dt))
        if dt>=SINGLE_DT: min_span=0.0; min_votes=1
    runs={}
    for a,t in zip(frames,times):
        sc,_=frame_scores(a,H,near,disks,big_px,rad_k)
        if blocked:
            sc={p:v for p,v in sc.items() if not any((p[0]==h or p[1]==h) and t0<t<t1 for h,t0,t1 in blocked)}
        items=sorted(sc.items(), key=lambda x:-x[1]); used=set()
        for p,hit in items:
            if p[0] in used or p[1] in used: continue
            used|=set(p)
            L=runs.setdefault(p,[])
            if L and t-L[-1][1]<=gap and t-L[-1][0]<max_span: L[-1][1]=t; L[-1][2]+=1; L[-1][3]+=hit
            else: L.append([t,t,1,hit])
    ev=[(p,r[0],r[1],r[2],r[3]) for p,L in runs.items() for r in L if r[2]>=min_votes and r[1]-r[0]>=min_span and (r[2]>=2 or r[3]>=SINGLE_MIN)]
    ev.sort(key=lambda e:-e[4]); keep=[]
    for e in ev:
        ok=True
        for k in keep:
            if set(e[0])&set(k[0]) and not (e[2]<k[1]-0.1 or e[1]>k[2]+0.1): ok=False;break
        if ok: keep.append(e)
    keep.sort(key=lambda e:e[1])
    return keep
