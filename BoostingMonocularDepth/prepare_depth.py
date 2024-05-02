import torch
# import BoostingMonocularDepth.run as get_depth
import os
import numpy as np
import sys
sys.path.append('BoostingMonocularDepth')
import run

def prepare_gt_depth(input_folder: str, save_folder: str):
    with torch.no_grad():
        os.makedirs(save_folder, exist_ok=True) 
        run.get_monodepth(input_folder, save_folder)


