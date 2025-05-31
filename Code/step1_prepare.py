import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from mp_utils.mp_utils_code import LMKExtractor
from mp_utils.draw_util import FaceMeshVisualizer


part2lmk = {'lips': [0, 267, 269, 270, 13, 14, 17, 402, 146, 405, 409, 415, 291, 37, 39, 40, 178, 308, 181, 310, 311, 312, 185, 314, 317, 318, 61, 191, 321, 324, 78, 80, 81, 82, 84, 87, 88, 91, 95, 375], \
                'left_eye': [384, 385, 386, 387, 388, 390, 263, 362, 398, 466, 373, 374, 249, 380, 381, 382], \
                'left_eyebrow': [293, 295, 296, 300, 334, 336, 276, 282, 283, 285], \
                'right_eyebrow': [65, 66, 70, 105, 107, 46, 52, 53, 55, 63], \
                'right_eye': [160, 33, 161, 163, 133, 7, 173, 144, 145, 246, 153, 154, 155, 157, 158, 159], \
                'left_iris': [474, 475, 476, 477] + [473], \
                'right_iris': [469, 470, 471, 472] + [468], \
                'nose': [1, 2, 4, 5, 6, 19, 275, 278, 294, 168, 45, 48, 440, 64, 195, 197, 326, 327, 344, 220, 94, 97, 98, 115]}


class FaceMeshDetector:
    def __init__(self) -> None:
        self.lmk_extractor = LMKExtractor()
        self.vis = FaceMeshVisualizer(forehead_edge=False, iris_edge=True, iris_point=True)
        self.mery_neutral_lmk = None
        self.mh_neutral_lmk = None
        
    def get_neutral(self, mery_img, mh_img):
        mery_img = cv2.imread(str(mery_img))
        mery_img = cv2.resize(mery_img, (512, 512))
        mery_rgb = cv2.cvtColor(mery_img, cv2.COLOR_BGR2RGB)
        mery_face_result = self.lmk_extractor(mery_rgb)
        mery_face_result['width'] = mery_rgb.shape[1]
        mery_face_result['height'] = mery_rgb.shape[0]
        mery_lmks = mery_face_result['lmks'].astype(np.float32)
        self.mery_neutral_lmks = mery_lmks

        mh_img = cv2.imread(str(mh_img))
        mh_img = cv2.resize(mh_img, (512, 512))
        mh_rgb = cv2.cvtColor(mh_img, cv2.COLOR_BGR2RGB)
        mh_face_result = self.lmk_extractor(mh_rgb)
        mh_face_result['width'] = mh_rgb.shape[1]
        mh_face_result['height'] = mh_rgb.shape[0]
        mh_lmks = mh_face_result['lmks'].astype(np.float32)
        self.mh_neutral_lmks = mh_lmks
        # mh_rotation = mh_face_result['matrix'][:3, :3]
        # self.mh_rotation = mh_rotation
   
    
    def brow_alignment(self, neutral_brow_lmks1, now_brow_lmks1, neutral_brow_lmks2): 
        ## part global alignment
        global_move = np.mean(now_brow_lmks1 - neutral_brow_lmks1, axis=0)
        
        ## part local alignment
        scale_x = (np.max(neutral_brow_lmks2[:, 0]) - np.min(neutral_brow_lmks2[:, 0])) / (np.max(neutral_brow_lmks1[:, 0]) - np.min(neutral_brow_lmks1[:, 0]) + 1e-6)
        scale_y = (np.max(neutral_brow_lmks2[:, 1]) - np.min(neutral_brow_lmks2[:, 1])) / (np.max(neutral_brow_lmks1[:, 1]) - np.min(neutral_brow_lmks1[:, 1]) + 1e-6)
        local_move_x = (now_brow_lmks1[:, 0] - (neutral_brow_lmks1[:, 0] + global_move[0])) * scale_x
        local_move_y = (now_brow_lmks1[:, 1] - (neutral_brow_lmks1[:, 1] + global_move[1])) * scale_y
        local_move = np.stack([local_move_x, local_move_y], axis=1)
        
        ## combine global and local alignment
        aligned_brow_lmks = neutral_brow_lmks2 + global_move + local_move
        
        return aligned_brow_lmks
        # return neutral_brow_lmks2 + (now_brow_lmks1 - neutral_brow_lmks1)
        
    
    def lips_alignment(self, neutral_lips_lmks1, now_lips_lmks1, neutral_lips_lmks2):
        ## part global alignment
        global_move = np.mean(now_lips_lmks1 - neutral_lips_lmks1, axis=0)
        
        ## part local alignment
        scale_x = (np.max(neutral_lips_lmks2[:, 0]) - np.min(neutral_lips_lmks2[:, 0])) / (np.max(neutral_lips_lmks1[:, 0]) - np.min(neutral_lips_lmks1[:, 0]) + 1e-6)
        scale_y = (np.max(neutral_lips_lmks2[:, 1]) - np.min(neutral_lips_lmks2[:, 1])) / (np.max(neutral_lips_lmks1[:, 1]) - np.min(neutral_lips_lmks1[:, 1]) + 1e-6)
        local_move_x = (now_lips_lmks1[:, 0] - (neutral_lips_lmks1[:, 0] + global_move[0])) * scale_x
        local_move_y = (now_lips_lmks1[:, 1] - (neutral_lips_lmks1[:, 1] + global_move[1])) * scale_y
        local_move = np.stack([local_move_x, local_move_y], axis=1)
        
        ## combine global and local alignment
        aligned_lips_lmks = neutral_lips_lmks2 + global_move + local_move
        
        return aligned_lips_lmks
        # return neutral_lips_lmks2 + (now_lips_lmks1 - neutral_lips_lmks1)
        
        
        
    def nose_alignment(self, neutral_nose_lmks1, now_nose_lmks1, neutral_nose_lmks2):
        ## part global alignment
        global_move = np.mean(now_nose_lmks1 - neutral_nose_lmks1, axis=0)
        
        ## part local alignment
        scale_x = (np.max(neutral_nose_lmks2[:, 0]) - np.min(neutral_nose_lmks2[:, 0])) / (np.max(neutral_nose_lmks1[:, 0]) - np.min(neutral_nose_lmks1[:, 0]) + 1e-6)
        scale_y = (np.max(neutral_nose_lmks2[:, 1]) - np.min(neutral_nose_lmks2[:, 1])) / (np.max(neutral_nose_lmks1[:, 1]) - np.min(neutral_nose_lmks1[:, 1]) + 1e-6)
        local_move_x = (now_nose_lmks1[:, 0] - (neutral_nose_lmks1[:, 0] + global_move[0])) * scale_x
        local_move_y = (now_nose_lmks1[:, 1] - (neutral_nose_lmks1[:, 1] + global_move[1])) * scale_y
        local_move = np.stack([local_move_x, local_move_y], axis=1)
        
        aligned_nose_lmks = neutral_nose_lmks2 + local_move
        
        return aligned_nose_lmks
        # return neutral_nose_lmks2 + (now_nose_lmks1 - neutral_nose_lmks1)
        
        

    def eye_alignment(self, neutral_eye_lmks1, now_eye_lmks1, neutral_eye_lmks2):
        ## part local alignment
        dis_x_1 = np.max(now_eye_lmks1[:, 0]) - np.min(now_eye_lmks1[:, 0])
        dis_x_2 = np.max(neutral_eye_lmks2[:, 0]) - np.min(neutral_eye_lmks2[:, 0])
        
        # dis_y_1 = np.max(now_eye_lmks1[:, 1]) - np.min(now_eye_lmks1[:, 1])
        dis_y_1 = now_eye_lmks1[:, 1] - np.mean(now_eye_lmks1[:, 1])
        
        ### dis_y_1 / dis_x_1 ==> dis_y_2 / dis_x_2
        dis_y_2 = dis_y_1 * (dis_x_2 / (dis_x_1 + 1e-6))
        
        ## combine global and local alignment
        aligned_eye_lmks = neutral_eye_lmks2
        aligned_eye_lmks[:, 1] = (dis_y_2 + np.mean(neutral_eye_lmks2[:, 1]))
        
        return aligned_eye_lmks
        
        
    
    def iris_alignment(self, now_eye_lmks1, now_iris_lmk1, now_eye_lmks2):
        ## part global alignment
        now_iris_lmk1_center = now_iris_lmk1[-1]
        dis_x_left = now_iris_lmk1_center[0] - np.min(now_eye_lmks1[:, 0])
        dis_x_right = np.max(now_eye_lmks1[:, 0]) - now_iris_lmk1_center[0]
        dis_y_top = now_iris_lmk1_center[1] - np.min(now_eye_lmks1[:, 1])
        dis_y_bottom = np.max(now_eye_lmks1[:, 1]) - now_iris_lmk1_center[1]
        
        ### dis_x_left / dis_x_right ==> (now_iris_lmk2_center[0] - np.min(now_eye_lmks2[:, 0])) / (np.max(now_eye_lmks2[:, 0]) - now_iris_lmk2_center[0])
        ### dis_y_top / dis_y_bottom ==> (now_iris_lmk2_center[1] - np.min(now_eye_lmks2[:, 1])) / (np.max(now_eye_lmks2[:, 1]) - now_iris_lmk2_center[1])
        now_iris_lmk2_center = np.zeros_like(now_iris_lmk1_center)
        now_iris_lmk2_center[0] = (dis_x_left * np.max(now_eye_lmks2[:, 0]) + dis_x_right * np.min(now_eye_lmks2[:, 0])) / (dis_x_left + dis_x_right)
        now_iris_lmk2_center[1] = (dis_y_top * np.max(now_eye_lmks2[:, 1]) + dis_y_bottom * np.min(now_eye_lmks2[:, 1])) / (dis_y_top + dis_y_bottom)
        
        aligned_iris_lmk = now_iris_lmk2_center + (now_iris_lmk1 - now_iris_lmk1_center)
        
        return aligned_iris_lmk
        
        
        
    def __call__(self, image_path):
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, (512, 512))
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_result = self.lmk_extractor(frame_rgb)
        face_result['width'] = frame_rgb.shape[1]
        face_result['height'] = frame_rgb.shape[0]
        mh_now_lmks = face_result['lmks'].astype(np.float32)
        
        
        ## Remove z-coordinate and keep only x, y
        mh_neutral_lmks = self.mh_neutral_lmks[:, :2]
        mery_neutral_lmks = self.mery_neutral_lmks[:, :2]
        mh_now_lmks = mh_now_lmks[:, :2]
        
        mery_now_lmks = mh_now_lmks - mh_neutral_lmks + mery_neutral_lmks
        
        
        ## Local Alignment
        ### Brow
        mery_now_lmks[part2lmk['left_eyebrow']] = self.brow_alignment(
                                                    neutral_brow_lmks1=mh_neutral_lmks[part2lmk['left_eyebrow']],
                                                    now_brow_lmks1=mh_now_lmks[part2lmk['left_eyebrow']],
                                                    neutral_brow_lmks2=mery_neutral_lmks[part2lmk['left_eyebrow']])
        mery_now_lmks[part2lmk['right_eyebrow']] = self.brow_alignment(
                                                    neutral_brow_lmks1=mh_neutral_lmks[part2lmk['right_eyebrow']],
                                                    now_brow_lmks1=mh_now_lmks[part2lmk['right_eyebrow']],
                                                    neutral_brow_lmks2=mery_neutral_lmks[part2lmk['right_eyebrow']])
        
        ### Lips
        mery_now_lmks[part2lmk['lips']] = self.lips_alignment(
                                                    neutral_lips_lmks1=mh_neutral_lmks[part2lmk['lips']],
                                                    now_lips_lmks1=mh_now_lmks[part2lmk['lips']],
                                                    neutral_lips_lmks2=mery_neutral_lmks[part2lmk['lips']])
        
        ### Nose
        mery_now_lmks[part2lmk['nose']] = self.nose_alignment(
                                                    neutral_nose_lmks1=mh_neutral_lmks[part2lmk['nose']],
                                                    now_nose_lmks1=mh_now_lmks[part2lmk['nose']],
                                                    neutral_nose_lmks2=mery_neutral_lmks[part2lmk['nose']])
        
        ### Eye
        mery_now_lmks[part2lmk['left_eye']] = self.eye_alignment(
                                                    neutral_eye_lmks1=mh_neutral_lmks[part2lmk['left_eye']],
                                                    now_eye_lmks1=mh_now_lmks[part2lmk['left_eye']],
                                                    neutral_eye_lmks2=mery_neutral_lmks[part2lmk['left_eye']])
        mery_now_lmks[part2lmk['right_eye']] = self.eye_alignment(
                                                    neutral_eye_lmks1=mh_neutral_lmks[part2lmk['right_eye']],
                                                    now_eye_lmks1=mh_now_lmks[part2lmk['right_eye']],
                                                    neutral_eye_lmks2=mery_neutral_lmks[part2lmk['right_eye']])

        ### Iris
        mery_now_lmks[part2lmk['left_iris']] = self.iris_alignment(
                                                    now_eye_lmks1=mh_now_lmks[part2lmk['left_eye']],
                                                    now_iris_lmk1=mh_now_lmks[part2lmk['left_iris']],
                                                    now_eye_lmks2=mery_now_lmks[part2lmk['left_eye']])
        mery_now_lmks[part2lmk['right_iris']] = self.iris_alignment(
                                                    now_eye_lmks1=mh_now_lmks[part2lmk['right_eye']],
                                                    now_iris_lmk1=mh_now_lmks[part2lmk['right_iris']],
                                                    now_eye_lmks2=mery_now_lmks[part2lmk['right_eye']])
        
        ## Draw Landmarks
        lmk_img = self.vis.draw_landmarks((frame_rgb.shape[1], frame_rgb.shape[0]), mery_now_lmks , normed=True)
        
        return lmk_img 


if __name__=='__main__':
    detector = FaceMeshDetector()
    detector.get_neutral(
        mery_img=Path("./ref_data/target_neutral.jpg"),
        mh_img=Path("./ref_data/mh_neutral.jpg")
    )
    
    src_dir = Path("./data/source")
    dst_dir = Path("./mid_data/lmk")
    
    for subdir in tqdm(src_dir.iterdir()):
        dst_subdir = dst_dir / subdir.name
        dst_subdir.mkdir(parents=True, exist_ok=True)
        for img_path in tqdm(subdir.iterdir()):
            dst_img_path = dst_subdir / img_path.name
            lmk_img = detector(img_path)
            lmk_img = cv2.cvtColor(lmk_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(dst_img_path, lmk_img)
    


        