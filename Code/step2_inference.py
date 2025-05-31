from tqdm import tqdm
import torch
import cv2
from collections import OrderedDict
from pathlib import Path
import numpy as np
from models.model import LMK2Rig

class Vid2RigRunner:
    def __init__(self):
        model_path = "./pretrained_retargeting.pth"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path)
        

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                k = k[7:]
            new_state_dict[k] = v
            
        model = LMK2Rig()
        model.load_state_dict(new_state_dict)
        model.eval()
        return model


    def predict_images(self, img_dir, rig_file):
        model = self.model
        
        image_files = sorted(img_dir.iterdir(), key=lambda p: p.name) 
        print(f"--Size: {len(image_files)}")
        img_lst = []
        
        for img_file in tqdm(image_files, desc="Predicting"):
            img = cv2.imread(img_file) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img.astype(np.float32) / 255.).permute(2, 0, 1) 
            img_lst.append(img_tensor)
            
        with torch.no_grad():
            imgs = torch.stack(img_lst)
            batches = torch.split(imgs, 128, dim=0)
            model = model.to(self.device)
            
            for batch in batches:
                batch = batch.to(self.device)
                output = model(batch) 
                self.write_to_rig_file(output.to("cpu"), rig_file, "a")


    def write_to_rig_file(self, data, file_path, mode="w"):
        with open(file_path, mode) as f:
            for row in data:
                row_str = ','.join([f"{x:.6f}" for x in row.tolist()])
                f.write(row_str + '\n')


    def __call__(self, img_dir, rig_dir):
        img_dir = Path(img_dir)
        rig_dir = Path(rig_dir)
        rig_dir.mkdir(parents=True, exist_ok=True)
        rig_path = rig_dir / (img_dir.stem + ".txt")
        self.predict_images(img_dir, rig_path)
            

if __name__=='__main__':
    
    all_dir = Path("./mid_data/lmk")
    output_dir = Path("./mid_data/rig")
    
    runner = Vid2RigRunner()
    for driving_dir in all_dir.iterdir():
        runner(driving_dir, output_dir)
        
        
    
    
    
                   
    


        