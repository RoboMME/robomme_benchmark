# repick controller checklist (v2)
- INIT: observe step 0 (main). Failure -> return (nothing possible).
- TRACK (track_video): pre: manifest with video_demo frames; grasp found from finger sum<0.07; N from early frames;
  start frame = first after release with EE z>0.12 and N clusters (fallback: frame-0 clusters + release point).
  Progress: logs min target separation / min points per object. Failure -> live-only fallback target.
- MATCH: mean-shift on live front points from track end (gate 5cm) else nearest cluster.
- PICK k (pick_place): wrist refine (mean-shift, gate 4cm) else front fallback; yaw chosen for fingertip clearance;
  grasp z 0.022; verify fingers>0.004 after close and after lift; 3 attempts with varied z/yaw; place at z 0.032, open.
  Main loop: up to n+2 calls, stops if <120 steps left.
- BUTTON: front low-sat bright raised blob, top-centroid; wrist refine gated (2cm, 1cm height); descend closed in 6mm
  increments to max(0.008, top-0.03) until episode ends; 2 retries.
- move_to: rotation-divergence guard (switch world->body deltas, then freeze).
- v2: scene_points excludes x<-0.45 (robot base) and points within 3.5cm of detected button; object count from
  frames with EE z>0.3 before grasp; tracks initialised from pre-grasp clusters with target moved to release point,
  tracking from first frame after release with EE z>0.12; saturation->height-only fallback; grasp z = top-0.019.
