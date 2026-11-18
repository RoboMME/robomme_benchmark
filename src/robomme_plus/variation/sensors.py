"""Low noise on public observations; physical cameras and task state stay exact."""
import hashlib
import numpy as np

DEPTH_SIGMA_MM=1.
DEPTH_DROPOUT=.0025
ROTATION_DEGREES=.25
TRANSLATION_MM=1.
FOCAL_RELATIVE_ERROR=.0025
PRINCIPAL_POINT_PIXELS=.25


class SensorNoise:
    def __init__(self,seed):
        self.seed=int(seed)
        self.counter=0
        self.biases={}
        for camera in ('front','wrist'):
            digest=hashlib.sha256(f'calibration-v1/{seed}/{camera}'.encode()).digest()
            rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
            axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
            x,y,z=axis
            skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
            theta=np.deg2rad(ROTATION_DEGREES)
            delta=np.eye(4)
            delta[:3,:3]=np.eye(3)+np.sin(theta)*skew+(1-np.cos(theta))*(skew@skew)
            direction=rng.normal(size=3)
            delta[:3,3]=direction/np.linalg.norm(direction)*TRANSLATION_MM/1000
            self.biases[camera]=dict(delta=delta,focal_sign=rng.choice([-1,1],size=2),
                                      principal_sign=rng.choice([-1,1],size=2))

    def transform(self,obs,info):
        obs,info=dict(obs),dict(info)
        for camera in ('front','wrist'):
            key=camera+'_depth_list'
            if key in obs:
                data=np.asarray(obs[key])
                digest=hashlib.sha256(f'depth-v1/{self.seed}/{camera}/{self.counter}'.encode()).digest()
                rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
                value=np.clip(np.rint(data.astype(float)+rng.normal(0,DEPTH_SIGMA_MM,data.shape)),1,32767)
                value[(data<=0)|(rng.random(data.shape)<DEPTH_DROPOUT)]=0
                obs[key]=value.astype(data.dtype)
            bias=self.biases[camera]
            key=camera+'_camera_extrinsic_list'
            if key in obs:
                data=np.asarray(obs[key])
                matrix=np.eye(4);matrix[:3]=data.reshape(3,4)
                obs[key]=(bias['delta']@matrix)[:3].reshape(data.shape).astype(data.dtype)
            key=camera+'_camera_intrinsic'
            if key in info:
                data=np.asarray(info[key]).copy();matrix=data.reshape(3,3)
                for axis in (0,1):
                    matrix[axis,axis]*=1+bias['focal_sign'][axis]*FOCAL_RELATIVE_ERROR
                    matrix[axis,2]+=bias['principal_sign'][axis]*PRINCIPAL_POINT_PIXELS
                info[key]=data
        self.counter+=1
        return obs,info
