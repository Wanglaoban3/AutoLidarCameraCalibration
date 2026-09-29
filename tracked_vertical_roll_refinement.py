"""Roll-only refinement using temporally and visually validated vertical tracks."""
import argparse, json
from pathlib import Path
import cv2, numpy as np, torch
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from scipy.optimize import least_squares
from nuscenes_edge_demo import bilinear, project, se3
from teed_edge_cache import TEEDCache
from teed_model import TEED

def sample(nusc, scene, offset):
    token = scene['first_sample_token']
    for _ in range(offset): token = nusc.get('sample', token)['next']
    return nusc.get('sample', token)

def frame(nusc, sample_record, tracks, dataroot, cache):
    sd=nusc.get('sample_data', sample_record['data']['CAM_FRONT']); cam=nusc.get('calibrated_sensor',sd['calibrated_sensor_token']); pose=nusc.get('ego_pose',sd['ego_pose_token'])
    image=cv2.imread(str(Path(dataroot)/sd['filename'])); K=np.asarray(cam['camera_intrinsic'],float); R=Quaternion(cam['rotation']).rotation_matrix
    points=[]
    for t in tracks:
        z=np.linspace(t['z']-1.5,t['z']+1.5,25); g=np.c_[np.full(25,t['xy'][0]),np.full(25,t['xy'][1]),z]
        ego=(g-np.asarray(pose['translation']))@Quaternion(pose['rotation']).rotation_matrix; points.append((ego-np.asarray(cam['translation']))@R)
    prob=cache.probability(image,sd['token']); threshold=np.percentile(prob,95); gx=cv2.Sobel(prob,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(prob,cv2.CV_32F,0,1,ksize=3)
    edge=(prob>=threshold)&(np.abs(gx)>=.75*np.hypot(gx,gy)); distance=cv2.distanceTransform((edge==0).astype(np.uint8),cv2.DIST_L2,3)
    T=np.eye(4); T[:3,:3],T[:3,3]=R,np.asarray(cam['translation'])
    return {'points':np.concatenate(points),'image':image,'shape':image.shape,'K':K,'distance':distance,'T':T}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataroot',required=True); p.add_argument('--initial-json',required=True); p.add_argument('--consistency-json',required=True); p.add_argument('--cache-dir',required=True); p.add_argument('--out',required=True); p.add_argument('--scene',type=int,default=0); p.add_argument('--train-offsets',type=int,nargs='+',default=[4,6,8]); p.add_argument('--holdout-offsets',type=int,nargs='+',default=[10,12]); p.add_argument('--max-roll-step-deg',type=float,default=1.0); a=p.parse_args()
    prior=json.loads(Path(a.initial_json).read_text()); consistency=json.loads(Path(a.consistency_json).read_text()); tracks=[x['track'] for x in consistency['tracks'] if x['accepted']]
    initial=np.asarray(prior['refined_body_correction'],float); manual=se3(np.asarray(prior['manual_body_noise'],float)); nusc=NuScenes(version='v1.0-mini',dataroot=a.dataroot,verbose=False); scene=nusc.scene[a.scene]
    model=TEED().to('cpu'); model.load_state_dict(torch.load('/workspace/models/teed_biped_epoch5.pth',map_location='cpu',weights_only=True),strict=True); model.eval(); cache=TEEDCache(a.cache_dir,model,torch.device('cpu'))
    def build(offsets):
        fs=[]
        for offset in offsets:
            f=frame(nusc,sample(nusc,scene,offset),tracks,a.dataroot,cache); f['manual']=np.linalg.inv(f['T'])@manual@f['T']; fs.append(f)
        return fs
    train,hold=build(a.train_offsets),build(a.holdout_offsets)
    def residual(fs,x):
        out=[]
        for f in fs:
            uv,valid=project(f['points'],np.linalg.inv(f['T'])@se3(np.r_[x,initial[1:]])@f['T']@f['manual'],f['K'],f['shape']); v=np.full(len(f['points']),30.); v[valid]=bilinear(f['distance'],uv[valid]); out.append(v)
        return np.concatenate(out)
    def score(fs,x):
        v=residual(fs,x); return {'mean_px':float(v.mean()),'median_px':float(np.median(v)),'p90_px':float(np.percentile(v,90)),'points':len(v)}
    before_train,before_hold=score(train,initial[0]),score(hold,initial[0]); step=np.deg2rad(a.max_roll_step_deg); r=least_squares(lambda x:residual(train,x[0]),[initial[0]],bounds=([initial[0]-step],[initial[0]+step]),loss='huber',f_scale=2.5,max_nfev=80); refined=initial.copy(); refined[0]=r.x[0]; after_train,after_hold=score(train,r.x[0]),score(hold,r.x[0]); expected=np.asarray(prior['expected_body_correction'],float)
    report={'tracks':len(tracks),'initial_body_correction':initial.tolist(),'refined_body_correction':refined.tolist(),'initial_error_rpy_deg':np.rad2deg(initial[:3]-expected[:3]).tolist(),'refined_error_rpy_deg':np.rad2deg(refined[:3]-expected[:3]).tolist(),'train_initial':before_train,'train_refined':after_train,'holdout_initial':before_hold,'holdout_refined':after_hold,'cache':{'hits':cache.hits,'inferred':cache.misses},'publish_roll':bool(r.success and after_hold['median_px']<before_hold['median_px'] and after_hold['p90_px']<before_hold['p90_px'])}
    Path(a.out).mkdir(parents=True,exist_ok=True); (Path(a.out)/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
