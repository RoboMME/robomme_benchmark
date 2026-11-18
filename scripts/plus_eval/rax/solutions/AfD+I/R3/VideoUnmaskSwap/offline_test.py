import json, sys, numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, '/workspace/code/vus')
from tracker import Tracker
m=json.load(open('/workspace/runtime/demonstrations/manifest.json'))
R='/workspace/runtime/demonstrations/'
step = float(sys.argv[1]) if len(sys.argv)>1 else 1.0
for e in range(3):
  d=m['demonstrations'][e]
  vid=[f for f in d['frames'] if f['phase']=='video_demo']
  idx=np.unique(np.round(np.arange(0,len(vid)-1+1e-9,step)).astype(int)).tolist()
  if idx[-1]!=len(vid)-1: idx.append(len(vid)-1)
  F=[np.asarray(Image.open(R+vid[i]['file']).convert('RGB'))[:, :256] for i in idx]
  t=Tracker(F); a=t.run()
  print(e, d['goal'][40:], 'N', t.N, 'cs', t.cont_start)
  im=Image.fromarray(F[-1]).resize((512,512)); dr=ImageDraw.Draw(im)
  for r in a:
    x,y=r['final']; print('   ',r['color'],r['rgb'],[round(v) for v in r['final']])
    col={'red':(255,0,0),'green':(0,255,0),'blue':(0,0,255)}.get(r['color'],(0,0,0))
    dr.ellipse([2*x-6,2*y-6,2*x+6,2*y+6],outline=col,width=3)
  im.save(f'/workspace/output/analysis/track_final_{e}.png')
  print('   min npx per track', np.array(t.npx).min(0))
