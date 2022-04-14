import glob 
import os 
from pathlib import Path


def run_model(video_path, image_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cmd = f"CUDA_VISIBLE_DEVICES=0 taskset -c 0-9 python demo.py \
            --config config/vox-adv-256.yaml  --checkpoint checkpoints/vox-adv-cpk.pth.tar --source_image {image_path} \
             --driving_video {video_path} --result_video {save_dir}/result.mp4 \
            --relative --frames --video"
    
    os.system(cmd) 





if __name__ == '__main__':
    image_dir = '/home/users/abhishekm/art-flow/golden-set/Images/*'
    video_dir = '/home/users/abhishekm/art-flow/golden-set/Videos_2/*'
    save_dir = '/home/users/abhishekm/art-flow/fomm-relative-videos-2-images-1'
    images = glob.glob(image_dir) 
    videos = glob.glob(video_dir) 

    for image_path in images:
        for video_path in videos:
            image_name = Path(image_path).stem 
            video_name = Path(video_path).stem 
            save_path = os.path.join(save_dir, video_name, image_name) 
            run_model(video_path, image_path, save_path)

