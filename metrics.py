#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import gc
from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir, exclude_dir, holdout_path, which_half):
    renders = []
    gts = []
    image_names = []

    exclude_fnames = [fname.split('.')[0] for fname in os.listdir(exclude_dir)]
    if holdout_path is not None:
        exclude_fnames.extend([fname.split('.')[0] for fname in os.listdir(holdout_path)])

    print(exclude_fnames)
    print('Render Dir:', renders_dir)
    
    render_fnames = os.listdir(renders_dir)
    split_point = len(render_fnames) // 2
    
    sel_range = None
    if which_half == 0:
        sel_range = range(0, split_point)
    elif which_half == 1:
        sel_range = range(split_point, len(render_fnames))
    
    for i in sel_range:
        fname = render_fnames[i]
        if fname.split('.')[0] in exclude_fnames:
            print('Skipping', fname)
            continue
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, exclude_path, holdout_path):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "renders"
            exclude_dir = Path(exclude_path) / "images"

            for method in os.listdir(test_dir):
                
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"
                renders_dir = method_dir / "renders"
                ssims = []
                psnrs = []
                lpipss = []
                image_namess = []

                for i in range(2):
                    print("Half:", i)
                    renders, gts, image_names = readImages(renders_dir, gt_dir, exclude_dir, holdout_path, i)
                    image_namess.extend(image_names)
                    

                    for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                        ssims.append(ssim(renders[idx], gts[idx]))
                        psnrs.append(psnr(renders[idx], gts[idx]))
                        lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                    # print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                    # print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                    # print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                    # print("")
                        
                    
                    del renders

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")
                
                print(f'{len(ssims)} images written')

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                            "PSNR": torch.tensor(psnrs).mean().item(),
                                                            "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_namess)},
                                                                "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_namess)},
                                                                "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_namess)}})
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print(e)
            print("Unable to compute metrics for model", scene_dir)

if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--exclude_path', '-e', type=str, default=None)
    parser.add_argument('--forceout', '-f', type=str, default=None)
    args = parser.parse_args()
    evaluate(args.model_paths, args.exclude_path, args.forceout)