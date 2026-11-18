import json, re, numpy as np
from pathlib import Path
from PIL import Image
from perception import blobs, cube_pos, find_visits
from swaps import detect_swaps2

USE_EXM=True; BLOCK=False; BIG_PX=0; RAD_K=0
WORDS={'1st':1,'2nd':2,'3rd':3,'4th':4,'5th':5,'6th':6,'first':1,'second':2,'third':3,'fourth':4,'fifth':5,'sixth':6,'seventh':7,'eighth':8,'last':-1}

def parse_instruction(ins):
    ins=ins.lower()
    color=None
    for c in ['red','green','blue']:
        if f'{c} cube' in ins: color=c
    n=None
    for w,v in WORDS.items():
        if re.search(r'\b'+w+r'\b target', ins): n=v
    return color, n

HMAX=0.03
EXM=[None]
_ZCACHE={}
def height_map(depth, f):
    K=np.array(f['camera_intrinsics']['front']['matrix']); pose=f['camera_poses']['front']
    from robot import quat_rot
    R=quat_rot(pose['quaternion_wxyz']); t=np.array(pose['position_m'])
    key=(tuple(K.ravel()),tuple(R.ravel()))
    if key not in _ZCACHE:
        h,w=depth.shape; v,u=np.mgrid[0:h,0:w]
        ray=np.stack([(u-K[0,2])/K[0,0], -(v-K[1,2])/K[1,1], -np.ones_like(u,dtype=float)],-1)
        _ZCACHE[key]=(ray@R.T)[...,2]
    return _ZCACHE[key]*depth + t[2]

def load_sequence(manifest_path, ws=Path('/workspace'), use_depth=True, log=print):
    m=json.loads((ws/manifest_path).read_text())
    F=[];T=[];ex=None;FM=[]
    for f in sorted(m['frames'], key=lambda f:f['frame_index']):
        files={(x['camera'],x['kind']):x['path'] for x in f['files']}
        rgb=np.array(Image.open(ws/files[('front','rgb')]).convert('RGB'))
        masked=rgb
        if use_depth and ('front','depth') in files and 'camera_poses' in f:
            try:
                dep=np.array(Image.open(ws/files[('front','depth')])).astype(np.float32)*0.001
                z=height_map(dep,f)
                masked=rgb.copy(); masked[(z>HMAX)|(dep<=0)]=0
            except Exception as e:
                log('depth mask failed',repr(e))
        if f['phase']=='video_demo':
            F.append(rgb); FM.append(masked); T.append(float(f['timestamp_s']))
        elif f['phase']=='execution':
            ex=rgb; EXM[0]=masked
    return F,T,ex,FM

def merge_visits(visits, T, gap=5.0):
    out=[]
    for v in visits:
        if out and out[-1][0]==v[0] and T[v[1]]-T[out[-1][2]]<gap: out[-1][2]=v[2]
        else: out.append(list(v))
    return out

def solve(F, T, ex_rgb, color, n, log=print, FM=None):
    if FM is None: FM=F
    src=EXM[0] if (EXM[0] is not None and EXM[0].shape==ex_rgb.shape and USE_EXM) else ex_rgb
    homes=[(x,y) for x,y,s in blobs(src) if 50<=s<=650]
    if len(homes)<2: homes=[(x,y) for x,y,s in blobs(ex_rgb) if s>=50]
    log(f'homes {[(round(x),round(y)) for x,y in homes]}')
    visits,_=find_visits(F,T,homes,color)
    visits=merge_visits(visits,T)
    blocked=[(h,T[s],T[e]) for h,s,e in visits] if BLOCK else None
    ev=detect_swaps2(FM,T,homes,big_px=BIG_PX,rad_k=RAD_K,blocked=blocked)
    perm=list(range(len(homes)))
    tl=[((s+e)/2,'swap',p) for p,s,e,_,_ in ev]+[((T[s]+T[e])/2,'visit',h) for h,s,e in visits]
    tl.sort(key=lambda x:x[0]); vids=[]
    for t,k,x in tl:
        if k=='swap': i,j=x; perm[i],perm[j]=perm[j],perm[i]
        else: vids.append(perm[x])
    log(f'swaps {len(ev)} visits {[v[0] for v in visits]} ids {vids}')
    ans=None
    if vids:
        k = n-1 if (n is not None and n>0) else len(vids)-1
        k = min(max(k,0), len(vids)-1)
        if n is not None and n>len(vids): log(f'WARN n={n} > visits {len(vids)}; using last')
        ans=perm.index(vids[k])
    return homes, ans, dict(n_swaps=len(ev), visits=[v[0] for v in visits], vids=vids)
