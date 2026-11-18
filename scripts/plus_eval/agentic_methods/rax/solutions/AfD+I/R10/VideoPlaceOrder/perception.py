import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

def target_mask(a):
    a=a.astype(int); R,G,B=a[...,0],a[...,1],a[...,2]
    return (B>G+25)&(B>100)&(R>70)

def blobs(a, minpx=25, maxpx=700):
    m=target_mask(a)
    lab,n=ndimage.label(ndimage.binary_closing(m,iterations=2))
    out=[]
    if n==0: return out
    idx=np.arange(1,n+1)
    sz=ndimage.sum(m,lab,idx)
    cm=ndimage.center_of_mass(m,lab,idx)
    for s,c in zip(sz,cm):
        if minpx<=s<=maxpx: out.append((c[1],c[0],s))
    return out

def cube_mask(a,color):
    a=a.astype(int); R,G,B=a[...,0],a[...,1],a[...,2]
    if color=='red': return (R>120)&(G<70)&(B<70)&(R>2*G)
    if color=='green': return (G>120)&(R<110)&(B<110)&(G>R+50)
    if color=='blue': return (B>110)&(R<70)&(G<70)&(B>R+60)
    raise ValueError(color)

def cube_pos(a,color):
    m=cube_mask(a,color)
    lab,n=ndimage.label(m)
    if n==0: return None
    sz=ndimage.sum(m,lab,np.arange(1,n+1)); j=int(np.argmax(sz))+1
    if sz[j-1]<15: return None
    ys,xs=np.nonzero(lab==j); return (xs.mean(),ys.mean(),int(sz[j-1]))

def track(frames_rgb, times, homes, gate_px_per_s=120.0, home_tol=6.0, log=None):
    """frames_rgb: list of front RGB arrays. homes: Nx2 array. Returns per-frame home->identity map list."""
    H=np.array(homes,float); N=len(H)
    pos=H.copy(); vel=np.zeros_like(H)
    ident_at_home=[list(range(N))]  # for frame 0
    perm=list(range(N))  # perm[home]=identity
    pos_id=H.copy()  # position of identity i
    last_t=times[0]
    hist=[]
    moving=False
    for k,(a,t) in enumerate(zip(frames_rgb,times)):
        dt=max(t-last_t,1e-3); last_t=t
        D=blobs(a)
        D=[d for d in D if d[2]>=40]
        pred=pos_id+vel*dt
        gate=max(8.0, gate_px_per_s*dt)
        if D:
            P=np.array([[d[0],d[1]] for d in D])
            C=np.linalg.norm(pred[:,None,:]-P[None,:,:],axis=2)
            Cg=np.where(C<gate,C,1e6)
            r,c=linear_sum_assignment(Cg)
            matched=set()
            for i,j in zip(r,c):
                if Cg[i,j]<1e6:
                    newp=P[j]
                    vel[i]=0.5*vel[i]+0.5*(newp-pos_id[i])/dt
                    pos_id[i]=newp; matched.add(i)
            for i in range(N):
                if i not in matched:
                    # occluded: if near home, hold; else coast slowly
                    dh=np.linalg.norm(H-pos_id[i],axis=1).min()
                    if dh<home_tol: vel[i]=0
                    else: pos_id[i]=pos_id[i]+vel[i]*dt*0.5; vel[i]*=0.5
        # snap: if every identity near some home -> rest state
        dist=np.linalg.norm(pos_id[:,None,:]-H[None,:,:],axis=2)
        r,c=linear_sum_assignment(dist)
        if dist[r,c].max()<home_tol:
            newperm=[None]*N
            for i,h in zip(r,c): newperm[h]=i
            if newperm!=perm and log is not None: log.append((k,t,'perm',perm,newperm))
            perm=newperm
            for i,h in zip(r,c): pos_id[i]=0.7*pos_id[i]+0.3*H[h]
        hist.append(list(perm))
    return hist

def find_visits(frames_rgb, times, homes, color, rest_s=2.0, tol=9.0):
    H=np.array(homes,float)
    seq=[]
    for a in frames_rgb:
        cp=cube_pos(a,color); seq.append(cp)
    # initial cube position
    p0=next((c for c in seq if c is not None),None)
    visits=[]  # (home, k_mid)
    cur=None; start=None
    def near_home(c):
        if c is None: return None
        q=np.array([c[0]+2.0,c[1]+3.0])
        d=np.linalg.norm(H-q,axis=1); h=int(np.argmin(d))
        return h if d[h]<tol else None
    runs=[]  # (home, k_start, k_end)
    for k,c in enumerate(seq):
        h=near_home(c)
        if c is not None and p0 is not None and np.hypot(c[0]-p0[0],c[1]-p0[1])<5: h=None
        if h is not None and runs and runs[-1][0]==h and k-runs[-1][2]<=max(3,int(0.0)+3) :
            runs[-1][2]=k
        elif h is not None:
            runs.append([h,k,k])
    # merge and filter by duration and stationarity
    out=[]
    for h,s,e in runs:
        if times[e]-times[s] >= rest_s:
            if out and out[-1][0]==h and times[s]-times[out[-1][2]]<2.0:
                out[-1][2]=e
            else: out.append([h,s,e])
    return out, seq

def pair_events(frames_rgb, times, homes, mid_tol=5.0, vac_tol=5.0, offhome=6.0):
    """Detect swaps: per frame, find home pairs whose two homes are vacant and two off-home detections are symmetric about their midpoint."""
    H=np.array(homes,float); N=len(H)
    pairs=[(i,j) for i in range(N) for j in range(i+1,N)]
    frame_pairs=[]
    for a in frames_rgb:
        D=[d for d in blobs(a) if d[2]>=40]
        P=np.array([[d[0],d[1]] for d in D]) if D else np.zeros((0,2))
        if len(P):
            dh=np.linalg.norm(P[:,None,:]-H[None,:,:],axis=2)
            vacant=[not (dh[:,h]<vac_tol).any() for h in range(N)]
            off=dh.min(1)>offhome
        else:
            vacant=[True]*N; off=np.zeros(0,bool)
        Q=P[off]
        found=[]
        for (i,j) in pairs:
            if not (vacant[i] and vacant[j]) or len(Q)<2: continue
            M=(H[i]+H[j])/2; sep=np.linalg.norm(H[i]-H[j])
            best=None
            for u in range(len(Q)):
                for v in range(u+1,len(Q)):
                    res=np.linalg.norm((Q[u]+Q[v])/2-M)
                    if res<mid_tol and np.linalg.norm(Q[u]-Q[v])>0.5*sep:
                        if best is None or res<best: best=res
            if best is not None: found.append((best,i,j))
        # greedy disjoint selection
        found.sort(); used=set(); sel=[]
        for res,i,j in found:
            if i in used or j in used: continue
            used|={i,j}; sel.append((i,j))
        frame_pairs.append(sel)
    # runs per pair
    events=[]
    active={}
    for k,sel in enumerate(frame_pairs):
        for p in sel:
            if p in active and times[k]-times[active[p][1]]<0.6: active[p][1]=k
            else:
                if p in active: events.append((p,)+tuple(active[p]))
                active[p]=[k,k]
    for p,v in active.items(): events.append((p,)+tuple(v))
    events=[(p,s,e) for p,s,e in events if times[e]-times[s]>=0.12 or e-s>=1]
    events.sort(key=lambda x:x[1])
    return events, frame_pairs

def resolve(events, visits, N, n, times):
    """Apply transpositions over time. perm[home]=identity."""
    perm=list(range(N))
    timeline=[]
    for p,s,e in events: timeline.append(((times[s]+times[e])/2,'swap',p))
    for h,s,e in visits: timeline.append(((times[s]+times[e])/2,'visit',h))
    timeline.sort(key=lambda x:x[0])
    vids=[]
    for t,kind,x in timeline:
        if kind=='swap':
            i,j=x; perm[i],perm[j]=perm[j],perm[i]
        else: vids.append(perm[x])
    ans=None
    if len(vids)>=n: ans=perm.index(vids[n-1])
    return ans, vids, perm, timeline
