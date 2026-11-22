"""VideoRepick controller: track the picked object through the conditioning video,
pick & place it twice, then press the button."""
import json, os, sys, time, base64, io, traceback
import numpy as np
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,HERE)
from geom import backproject
from percep import object_points, occupancy, blobs, X0, Y0, CELL, set_button, SAT, frame_points
from geom import load_rgb, load_depth
from track import track, occ_points
from motion import Robot, meta

LOGF=None
def log(*a):
    s=' '.join(str(x) for x in a)
    print(s, flush=True)
    try:
        with open('/workspace/output/controller_log.txt','a') as f: f.write(s+'\n')
    except Exception: pass

def decode_obs(reply):
    m=meta(reply); blocks=reply['content'] if isinstance(reply,dict) else reply
    imgs={}
    for a in m.get('attachments',[]):
        b=blocks[a['content_index']]
        arr=np.array(Image.open(io.BytesIO(base64.b64decode(b['data']))))
        imgs[(a['camera'],a['kind'])]=arr
    return m, imgs

def obs_points(m, imgs, cams=('front','wrist'), **kw):
    pts=[]
    for cam in cams:
        if (cam,'depth') not in imgs: continue
        rgb=imgs[(cam,'rgb')][...,:3].astype(np.int32); d=imgs[(cam,'depth')].astype(np.float64)*0.001
        pts.append(object_points(rgb,d,m['camera_intrinsics'][cam]['matrix'],m['camera_poses'][cam],**kw))
    return pts

def button_xy(m, imgs):
    """Button plate: dense flat patch at z~0.016 (colour-agnostic)."""
    import cv2
    occ=np.zeros((NYb,NXb),np.uint8)
    for cam in ('front','wrist'):
        if (cam,'depth') not in imgs: continue
        d=imgs[(cam,'depth')].astype(np.float64)*0.001
        P=backproject(d,m['camera_intrinsics'][cam]['matrix'],m['camera_poses'][cam])
        sel=(d>0)&(np.abs(P[...,2]-0.0160)<0.0035)&(np.abs(P[...,0])<0.38)&(np.abs(P[...,1])<0.38)
        pts=P[sel]
        ix=((pts[:,0]+0.4)/0.004).astype(int); iy=((pts[:,1]+0.4)/0.004).astype(int)
        occ[iy,ix]=1
    o=cv2.morphologyEx(occ,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    n,lab,st,cen=cv2.connectedComponentsWithStats(o)
    best=None
    for i in range(1,n):
        if st[i,4]>=40 and (best is None or st[i,4]>st[best,4]): best=i
    if best is None: return button_xy_color(m,imgs)
    w,h=st[best,2],st[best,3]
    return np.array([-0.4+(cen[best,0]+0.5)*0.004, -0.4+(cen[best,1]+0.5)*0.004])

NXb=NYb=200
def button_xy_color(m, imgs):
    res=[]
    for cam in ('front','wrist'):
        if (cam,'depth') not in imgs: continue
        rgb=imgs[(cam,'rgb')][...,:3].astype(np.int32); d=imgs[(cam,'depth')].astype(np.float64)*0.001
        P=backproject(d,m['camera_intrinsics'][cam]['matrix'],m['camera_poses'][cam])
        s=rgb.max(-1)-rgb.min(-1); br=rgb.mean(-1)
        sel=(d>0)&(s<30)&(br>150)&(P[...,2]>0.003)&(P[...,2]<0.05)&(np.abs(P[...,0])<0.35)&(np.abs(P[...,1])<0.35)
        if sel.sum()>=15:
            pts=P[sel][:,:2]
            # keep the dominant cluster
            med=np.median(pts,0); pts=pts[np.linalg.norm(pts-med,axis=1)<0.05]
            res.append((np.median(pts,0),len(pts)))
    if not res: return None
    w=np.array([n for _,n in res],float); xy=np.array([p for p,_ in res])
    return (xy*w[:,None]).sum(0)/w.sum()

def frame_imgs(fr):
    imgs={}
    for f in fr['files']:
        p='/workspace/'+f['path']
        if f['kind']=='rgb': imgs[(f['camera'],'rgb')]=load_rgb(p)
        else: imgs[(f['camera'],'depth')]=np.array(Image.open(p))
    return imgs

def button_from_frames(frames, m, imgs):
    cands=[]
    b=button_xy(m,imgs)
    if b is not None: cands.append(b)
    n=len(frames)
    for i in [0,n//4,n//2,3*n//4,n-2]:
        try:
            fr=frames[i]; b=button_xy(fr,frame_imgs(fr))
            if b is not None: cands.append(b)
        except Exception as e: log('button frame err',i,repr(e))
    if not cands: return None
    c=np.array(cands); med=np.median(c,0)
    log('button cands',np.round(c,3).tolist())
    return med

def refine_target(robot, guess, others, radius=0.035):
    """Observe and return refined XY of the object nearest guess, plus top z and neighbor list."""
    rep=robot.ctx.call('robomme_observe'); m,imgs=decode_obs(rep); robot._upd(m)
    pts=np.concatenate([p for p in obs_points(m,imgs) if len(p)] or [np.zeros((0,3))])
    if len(pts)==0: return guess, 0.04, []
    xy=pts[:,:2]
    # k-means with centers: guess + others (to split merged blobs)
    C=np.array([guess]+[o for o in others if np.linalg.norm(np.array(o)-guess)<0.12]).reshape(-1,2)
    for it in range(6):
        D=np.linalg.norm(xy[:,None]-C[None],axis=2); a=D.argmin(1); dm=D.min(1)
        for k in range(len(C)):
            sel=(a==k)&(dm<0.04)
            if sel.sum()>=5: C[k]=xy[sel].mean(0)
    D=np.linalg.norm(xy[:,None]-C[None],axis=2); a=D.argmin(1); dm=D.min(1)
    sel=(a==0)&(dm<0.04)
    top=float(np.percentile(pts[sel,2],95)) if sel.sum()>5 else 0.04
    new=C[0]
    if np.linalg.norm(new-guess)>0.03:
        log('refine jump too big', guess, new); new=np.array(guess)
    return new, top, [c for c in C[1:]]

def yaw_for(target, neighbors):
    """Return desired change in yaw (rad) so finger closing axis is perpendicular to nearest neighbour."""
    if not neighbors: return 0.0
    d=[np.linalg.norm(np.array(n)-target) for n in neighbors]
    j=int(np.argmin(d))
    if d[j]>0.07: return 0.0
    v=np.array(neighbors[j])-target
    ang=np.arctan2(v[1],v[0])  # neighbour direction
    # default finger axis = world y (angle +90deg). want axis perpendicular to v: axis angle = ang+90
    want=ang+np.pi/2; cur=np.pi/2
    dy=(want-cur+np.pi/2)%np.pi-np.pi/2
    return float(dy)

def rotate_yaw(robot, dyaw, grip):
    n=int(np.ceil(abs(dyaw)/0.15))
    if n==0: return
    step=dyaw/n/0.5
    robot.step([[0,0,0,0,0,step,grip]]*n)
    robot.hold(grip,3)

def grasp_cycle(robot, xy, top, place_xy):
    zg=float(np.clip(top-0.02,0.012,0.03))
    robot.move_to([xy[0],xy[1],0.12],-1,max_steps=80)
    robot.move_to([xy[0],xy[1],zg],-1,max_steps=50,vmax=0.4)
    robot.hold(1,12)
    g=robot.state[6]+robot.state[7]
    log('grasp fingers', g, 'z', robot.state[2])
    if g<0.006:
        return False
    robot.move_to([xy[0],xy[1],0.12],1,max_steps=50)
    g2=robot.state[6]+robot.state[7]
    log('after lift fingers', g2)
    if g2<0.006:
        return False
    robot.move_to([place_xy[0],place_xy[1],zg+0.006],1,max_steps=50,vmax=0.4)
    robot.hold(-1,8)
    robot.move_to([place_xy[0],place_xy[1],0.12],-1,max_steps=50)
    return True

def press_button(robot, bxy):
    robot.move_to([bxy[0],bxy[1],0.10],1,max_steps=80)
    for z in [0.06,0.045,0.035,0.028,0.022,0.016,0.010]:
        robot.move_to([bxy[0],bxy[1],z],1,max_steps=15,vmax=0.3)
        log('press z',z,robot.state[:3],robot.ended,robot.complete)
        if robot.ended: return

def main(ctx):
    try:
        _main(ctx)
    except Exception as e:
        log('MAIN EXCEPTION', repr(e), traceback.format_exc())

def _main(ctx):
    t0=time.time()
    os.makedirs('/workspace/output',exist_ok=True)
    robot=Robot(ctx,log=log)
    rep=ctx.call('robomme_observe'); m,imgs=decode_obs(rep); robot._upd(m)
    log('start state',robot.state, 'remaining', robot.remaining)
    seq=m.get('frame_sequence') or {}
    mp=seq.get('manifest_path')
    target=None; others=[]; bxy=None
    try:
        man=json.load(open(os.path.join('/workspace',mp)))
        frames=sorted(man['frames'],key=lambda f:f['frame_index'])
        bxy=button_from_frames(frames,m,imgs); set_button(bxy); log('button',bxy)
        b0,_=blobs(occupancy(frame_points(frames[0])))
        if len(b0)<2:
            SAT[0]=0; b1,_=blobs(occupancy(frame_points(frames[0])))
            log('colour-free fallback: blobs',len(b0),'->',len(b1))
            if len(b1)<2: SAT[0]=35
        r=track(frames)
        target=np.array(r['target_xy']); others=[np.array(p) for i,p in enumerate(r['final']) if i!=r['target']]
        log('track', {k:v for k,v in r.items() if k!='hist'}, 'time', time.time()-t0)
    except Exception as e:
        log('TRACK FAILED', repr(e), traceback.format_exc())
    if bxy is None:
        bxy=button_xy(m,imgs); set_button(bxy)
    log('button', bxy)
    # current-frame blobs
    b,_=blobs(occupancy(obs_points(m,imgs)))
    cur=[np.array([x,y]) for x,y,c in b]
    log('current blobs', [(round(x,3),round(y,3),c) for x,y,c in b])
    if target is None:
        if not cur: log('no objects'); return
        target=cur[0]; others=cur[1:]
    done=0; attempts=0
    place_xy=None
    while done<2 and attempts<5 and not robot.ended:
        attempts+=1
        robot.move_to([target[0],target[1],0.15],-1,max_steps=80)
        xy,top,nb=refine_target(robot,target,others)
        log('refined',xy,'top',top,'nbrs',[np.round(n,3) for n in nb])
        dyaw=yaw_for(xy,nb)
        if abs(dyaw)>0.05:
            # keep yaw at default for simplicity unless very close neighbor
            log('yaw adjust',dyaw)
            rotate_yaw(robot,dyaw,-1)
        if place_xy is None: place_xy=xy.copy()
        ok=grasp_cycle(robot,xy,top,place_xy)
        if abs(dyaw)>0.05:
            rotate_yaw(robot,-dyaw,-1)
        if ok: done+=1; target=place_xy
        else:
            log('grasp failed; retry')
            robot.hold(-1,6); robot.move_to([xy[0],xy[1],0.15],-1,max_steps=40)
            target=xy
        log('cycles done',done,'steps remaining',robot.remaining)
    if robot.ended:
        log('ended early', robot.complete); return
    if bxy is None:
        rep=ctx.call('robomme_observe'); m,imgs=decode_obs(rep); bxy=button_xy(m,imgs)
    if bxy is not None:
        press_button(robot,bxy)
    log('final ended',robot.ended,'complete',robot.complete,'time',time.time()-t0)
