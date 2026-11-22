import numpy as np, cv2
from geom import backproject, load_depth, load_rgb

CELL=0.004
X0,X1,Y0,Y1=-0.40,0.40,-0.40,0.40
NX=int((X1-X0)/CELL); NY=int((Y1-Y0)/CELL)

BUTTON=[None]
def set_button(xy): BUTTON[0]=None if xy is None else np.array(xy,float)

SAT=[35]
def object_points(rgb, depth, K, pose, zmin=0.008, zmax=0.09, sat=None):
    if sat is None: sat=SAT[0]
    P=backproject(depth,K,pose)
    z=P[...,2]
    s=rgb.max(-1)-rgb.min(-1)
    valid=(depth>0)&(z>zmin)&(z<zmax)&(s>sat)
    pts=P[valid]
    if sat<=0:
        # colour-free mode: drop points under tall structures (robot, occluders)
        tall=P[(depth>0)&(z>=zmax)&(z<0.6)]
        if len(tall) and len(pts):
            tc=set(zip(((tall[:,0]-X0)/0.02).astype(int),((tall[:,1]-Y0)/0.02).astype(int)))
            keys=zip(((pts[:,0]-X0)/0.02).astype(int),((pts[:,1]-Y0)/0.02).astype(int))
            keep=np.array([k not in tc for k in keys])
            pts=pts[keep]
    b=BUTTON[0]
    if b is not None and len(pts):
        inb=(np.abs(pts[:,0]-b[0])<0.045)&(np.abs(pts[:,1]-b[1])<0.045)&(pts[:,2]<0.036)
        pts=pts[~inb]
    return pts

HIGH=0.024
def occupancy(points_list):
    occ=np.zeros((NY,NX),np.uint8)
    for pts in points_list:
        if len(pts)==0: continue
        ix=((pts[:,0]-X0)/CELL).astype(int); iy=((pts[:,1]-Y0)/CELL).astype(int)
        ok=(ix>=0)&(ix<NX)&(iy>=0)&(iy<NY)
        occ[iy[ok],ix[ok]]=1
        hi=ok&(pts[:,2]>HIGH)
        occ[iy[hi],ix[hi]]=2
    return occ

def blobs(occ, min_cells=6):
    k=np.ones((3,3),np.uint8)
    o=cv2.morphologyEx((occ>0).astype(np.uint8),cv2.MORPH_CLOSE,k)
    n,lab,st,cen=cv2.connectedComponentsWithStats(o)
    out=[]
    for i in range(1,n):
        if st[i,4]>=min_cells and ((lab==i)&(occ==2)).sum()>=2:
            cx=X0+(cen[i,0]+0.5)*CELL; cy=Y0+(cen[i,1]+0.5)*CELL
            out.append((cx,cy,int(st[i,4])))
    return out,lab

def frame_points(fr, root='/workspace', cams=('front','wrist')):
    files={(f['camera'],f['kind']):f['path'] for f in fr['files']}
    pts=[]
    for cam in cams:
        if (cam,'depth') not in files: continue
        rgb=load_rgb(root+'/'+files[(cam,'rgb')]); d=load_depth(root+'/'+files[(cam,'depth')])
        pts.append(object_points(rgb,d,fr['camera_intrinsics'][cam]['matrix'],fr['camera_poses'][cam]))
    return pts
