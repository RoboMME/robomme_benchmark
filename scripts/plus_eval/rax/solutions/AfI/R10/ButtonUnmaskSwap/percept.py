import numpy as np
from scipy import ndimage
from geom import deproject, hsv_img

GRID=0.005; X0,X1=-0.45,0.25; Y0,Y1=-0.40,0.40
NX=int(round((X1-X0)/GRID)); NY=int(round((Y1-Y0)/GRID))

def world_points(rgb, depth, K, pose):
    P=deproject(depth,K,pose)
    h,s,v=hsv_img(rgb)
    return P,h,s,v

def white_mask(P,s,v,zmin=0.012,zmax=0.10):
    return (s<0.18)&(v>0.45)&(P[...,2]>zmin)&(P[...,2]<zmax)&(P[...,0]>X0)&(P[...,0]<X1)&(P[...,1]>Y0)&(P[...,1]<Y1)

def occupancy(points):
    g=np.zeros((NX,NY),np.float32)
    if len(points)==0: return g
    ix=((points[:,0]-X0)/GRID).astype(int); iy=((points[:,1]-Y0)/GRID).astype(int)
    ok=(ix>=0)&(ix<NX)&(iy>=0)&(iy<NY)
    np.add.at(g,(ix[ok],iy[ok]),1)
    return g

def detect(cams_pts, exclude=(), min_cells=6):
    """cams_pts: list of Nx3 arrays of white points. exclude: list of (x,y,r). Returns list of (xy, ncells)."""
    pts=np.concatenate([p for p in cams_pts if len(p)]) if any(len(p) for p in cams_pts) else np.zeros((0,3))
    for (ex,ey,r) in exclude:
        if len(pts):
            pts=pts[(pts[:,0]-ex)**2+(pts[:,1]-ey)**2>r*r]
    g=occupancy(pts)
    occ=ndimage.binary_closing(g>0,iterations=1)
    lab,n=ndimage.label(occ)
    out=[]
    for i in range(1,n+1):
        cells=np.argwhere(lab==i)
        if len(cells)<min_cells: continue
        c=cells.mean(0)
        out.append((np.array([X0+(c[0]+0.5)*GRID, Y0+(c[1]+0.5)*GRID]), len(cells)))
    return out, lab

def box_yaw(pts_xy):
    """Yaw (rad, in [-pi/4, pi/4)) of a square footprint from xy points via min-area bounding box."""
    if len(pts_xy)<10: return None, None
    c=pts_xy.mean(0); q=pts_xy-c
    best=None
    for a in np.deg2rad(np.arange(-45,45,1.0)):
        ca,sa=np.cos(a),np.sin(a)
        r=q@np.array([[ca,-sa],[sa,ca]])
        lo=np.percentile(r,3,axis=0); hi=np.percentile(r,97,axis=0)
        area=np.prod(hi-lo)
        if best is None or area<best[0]: best=(area,a,hi-lo)
    return best[1], best[2]

def container_top_points(P,s,v,center,radius=0.045,zlo=0.06,zhi=0.085):
    m=(s<0.2)&(v>0.4)&(P[...,2]>zlo)&(P[...,2]<zhi)
    pts=P[m]
    sel=np.linalg.norm(pts[:,:2]-np.asarray(center)[None],axis=1)<radius
    return pts[sel]

COLOR_NAMES=['red','green','blue','yellow','orange','purple','black','white','pink','cyan','brown','gray','grey']
def classify_pixels(h,s,v):
    lab=np.full(h.shape,'',dtype=object)
    sat=(s>0.45)&(v>0.3)
    lab[sat&((h<15)|(h>=340))]='red'
    lab[sat&(h>=15)&(h<50)]='orange'
    lab[sat&(h>=50)&(h<75)]='yellow'
    lab[sat&(h>=75)&(h<165)]='green'
    lab[sat&(h>=165)&(h<200)]='cyan'
    lab[sat&(h>=200)&(h<260)]='blue'
    lab[(s>0.3)&(v>0.3)&(h>=260)&(h<340)]='purple'
    lab[(v<0.3)]='black'
    lab[(s<0.12)&(v>0.8)]='white'
    lab[(s<0.1)&(v>0.5)&(v<=0.8)]='gray'
    return lab

def detect_cubes_buttons(P,h,s,v):
    """At step 0: cubes (colored, z 0.008-0.06) and grey button caps. Returns dict color->list of (xy,n), buttons list of xy."""
    lab=classify_pixels(h,s,v)
    obj=(P[...,2]>0.008)&(P[...,2]<0.07)&(P[...,0]>X0)&(P[...,0]<X1)&(P[...,1]>Y0)&(P[...,1]<Y1)
    res={}
    for col in ['red','green','blue','yellow','orange','purple','black','white','cyan','gray']:
        m=obj&(lab==col)
        if m.sum()<5: continue
        L,n=ndimage.label(m)
        for i in range(1,n+1):
            sel=L==i
            if sel.sum()<5: continue
            pts=P[sel]
            res.setdefault(col,[]).append((np.median(pts[:,:2],0), int(sel.sum()), float(np.percentile(pts[:,2],95))))
    return res

def cells_of(points, dilate=2):
    g=occupancy(points)>0
    if dilate: g=ndimage.binary_dilation(g,iterations=dilate)
    return g

def detect2(pts_list, static_mask=None, ee=None, ee_r=0.075, min_cells=25):
    pts=[p for p in pts_list if len(p)]
    pts=np.concatenate(pts) if pts else np.zeros((0,3))
    if ee is not None and len(pts):
        pts=pts[(pts[:,0]-ee[0])**2+(pts[:,1]-ee[1])**2>ee_r*ee_r]
    g=occupancy(pts)>0
    if static_mask is not None: g&=~static_mask
    g=ndimage.binary_closing(g,iterations=1)
    lab,n=ndimage.label(g)
    out=[]
    for i in range(1,n+1):
        cells=np.argwhere(lab==i)
        if len(cells)<min_cells: continue
        c=cells.mean(0)
        out.append((np.array([X0+(c[0]+0.5)*GRID, Y0+(c[1]+0.5)*GRID]), len(cells)))
    return out

def upper_white(P,s,v,zlo=0.045,zhi=0.095):
    m=(s<0.2)&(v>0.45)&(P[...,2]>zlo)&(P[...,2]<zhi)&(P[...,0]>X0)&(P[...,0]<X1)&(P[...,1]>Y0)&(P[...,1]<Y1)
    return P[m]
