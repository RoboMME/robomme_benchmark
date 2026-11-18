import json, sys, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, '/workspace/code/vus')
from tracker import Tracker, classify
from scipy import ndimage
m=json.load(open('/workspace/runtime/demonstrations/manifest.json'))
R='/workspace/runtime/demonstrations/'
step = float(sys.argv[1]) if len(sys.argv)>1 else 1.0
NEX = int(sys.argv[2]) if len(sys.argv)>2 else 100
def strict(a):
    a=a.astype(int); r,g,b=a[...,0],a[...,1],a[...,2]
    out={}
    for name,mk in [('red',(r>140)&(g<50)&(b<50)),('green',(g>r+60)&(g>b+60)),('blue',(b>r+60)&(b>g+60))]:
        lab,n=ndimage.label(mk); res=[]
        for i in range(1,n+1):
            ys,xs=np.nonzero(lab==i)
            if len(xs)>=8: res.append((xs.mean(),ys.mean()))
        out[name]=res
    return out
tot=0; good=0
for e in range(3):
  d=m['demonstrations'][e]
  vid=[f for f in d['frames'] if f['phase']=='video_demo']
  ex=[f for f in d['frames'] if f['phase']=='expert_execution']
  idx=np.unique(np.round(np.arange(0,len(vid)-1+1e-9,step)).astype(int)).tolist()
  if idx[-1]!=len(vid)-1: idx.append(len(vid)-1)
  F=[np.asarray(Image.open(R+vid[i]['file']).convert('RGB'))[:, :256] for i in idx]
  F+=[np.asarray(Image.open(R+ex[i]['file']).convert('RGB'))[:, :256] for i in range(NEX)]
  t=Tracker(F, is_exec=[False]*len(idx)+[True]*NEX); a=t.run()
  fin=np.array([r['final'] for r in a]); cols=[r['color'] for r in a]
  # revealed cubes in execution
  seen=set()
  for i in range(NEX, len(ex)):
    A=np.asarray(Image.open(R+ex[i]['file']).convert('RGB'))[:, :256]
    for name,lst in strict(A).items():
      for (x,y) in lst:
        key=(name, round(x/8), round(y/8))
        if any(s[0]==name and abs(s[1]-x)<8 and abs(s[2]-y)<8 for s in seen): continue
        seen.add((name,x,y,i))
        dd=np.linalg.norm(fin-np.array([x,y-4]),axis=1); j=dd.argmin()
        ok = cols[j]==name
        tot+=1; good+=ok
        print(e,'exec',i,name,'at',round(x),round(y),'nearest track',j,cols[j],'dist',round(dd[j],1), 'OK' if ok else 'XX', 'dists', np.round(np.sort(dd)[:3],1))
print('good',good,'/',tot)
