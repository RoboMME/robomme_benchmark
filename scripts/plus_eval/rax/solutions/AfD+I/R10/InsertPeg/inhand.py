import json, numpy as np, sys
from PIL import Image
sys.path.insert(0,'/workspace/code/insertpeg')
from rlib import quat_to_R, rotvec_to_R
def load(frame):
    files={(f['camera'],f['kind']):f['path'] for f in frame['files']}
    rgb=np.array(Image.open('/workspace/'+files[('wrist','rgb')]).convert('RGB'))
    d=np.array(Image.open('/workspace/'+files[('wrist','depth')])).astype(float)*0.001
    return rgb,d
def peg_points(frame, rowmax=185, dmax=0.1):
    rgb,d=load(frame)
    K=np.array(frame['camera_intrinsics']['wrist']['matrix'])
    v,u=np.nonzero((d>0)&(d<dmax))
    keep=v<rowmax; u=u[keep]; v=v[keep]; z=d[v,u]
    x=(u-K[0,2])*z/K[0,0]; y=-(v-K[1,2])*z/K[1,1]
    P=np.stack([x,y,-z],1)  # OpenGL camera frame
    cp=frame['camera_poses']['wrist']; Rc=quat_to_R(cp['quaternion_wxyz']); tc=np.array(cp['position_m'])
    Pw=P@Rc.T+tc
    s=np.array(frame['state']); Re=rotvec_to_R(s[3:6]); pe=s[:3]
    Pe=(Pw-pe)@Re   # in EE frame
    return Pe
def describe(Pe):
    c=Pe.mean(0); U,S,Vt=np.linalg.svd(Pe-c); ax=Vt[0]
    if ax[np.argmax(abs(ax))]<0: ax=-ax
    return c, ax, len(Pe), Pe.min(0), Pe.max(0)
if __name__=='__main__':
    M=sys.argv[1]; i=int(sys.argv[2])
    m=json.load(open(M)); fr={f['frame_index']:f for f in m['frames']}
    f=fr[i] if i>=0 else m['frames'][-1]
    np.set_printoptions(precision=4,suppress=True)
    print(describe(peg_points(f)))

def axis_point(Pe, xref=0.05):
    c, ax, n, mn, mx = describe(Pe)
    return c + ax * (xref - c[0]) / ax[0], ax, n

def rot_between(a, b):
    """R with R a = b"""
    from rlib import rotvec_to_R
    a = a/np.linalg.norm(a); b = b/np.linalg.norm(b)
    v = np.cross(a, b); s = np.linalg.norm(v); c = float(np.dot(a, b))
    if s < 1e-9: return np.eye(3)
    return rotvec_to_R(v / s * np.arctan2(s, c))

def correction(frame_video, frame_live):
    """returns (Rd, cv, cl): live EE target = video pose corrected"""
    pv, av, nv = axis_point(peg_points(frame_video))
    pl, al, nl = axis_point(peg_points(frame_live))
    Rd = rot_between(al, av)
    return Rd, pv, pl, nv, nl

def corrected_target(Sv, Rd, pv, pl):
    from rlib import rotvec_to_R, R_to_rotvec
    Rv = rotvec_to_R(Sv[3:6]); Rn = Rv @ Rd
    p = Sv[:3] + Rv @ pv - Rn @ pl
    out = np.array(Sv, float).copy(); out[:3] = p; out[3:6] = R_to_rotvec(Rn)
    return out
