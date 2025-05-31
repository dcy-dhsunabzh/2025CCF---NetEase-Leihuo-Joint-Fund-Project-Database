import maya.cmds as cmds
import maya.mel as mel
import os
import random


def apply_animation_from_txt(start=0):
    
    name_list = []
    name_path = "mery_rig.txt"
    data_path = "result_rig/1.txt"
    
    with open(name_path, 'r') as file:
        for line in file:
            name_list.append(line.strip())
            
    with open(data_path, 'r') as file:
        frame_num = -1
        for line in file:
            frame_num += 1
            data_list = line.strip().split(',')
            now_frame = start + frame_num
            cmds.currentTime(now_frame)
            
            for name, data in zip(name_list, data_list):
                data = float(data)
                if name in ["Mery_ac_rg_eye-translateX", "Mery_ac_rg_eye-translateY", "Mery_ac_lf_eye-translateX", "Mery_ac_lf_eye-translateY"]:
                    data = data * 20
                control_name = name.split('-')[0]
                attribute_name = name.split('-')[1]
                cmds.setAttr(f"{control_name}.{attribute_name}", data)
                cmds.setKeyframe(control_name, attribute=attribute_name, t=now_frame)
                
    return now_frame + 1

apply_animation_from_txt()