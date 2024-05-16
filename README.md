# SparseGS: Real-Time 360° Sparse View Synthesis using Gaussian Splatting



[![Paper](https://img.shields.io/badge/cs.CV-Paper-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2312.00206)
[![Project Page](https://img.shields.io/badge/SparseGS-Website-blue?logo=googlechrome&logoColor=blue)](https://formycat.github.io/SparseGS-Real-Time-360-Sparse-View-Synthesis-using-Gaussian-Splatting/)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FForMyCat%2FSparseGS&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)



---------------------------------------------------
<p align="center" >
  <img src="vid_garden.gif" alt="Ground Truth" width="100%" />
</p>

## Environment Setups
Conda environment:
```bash
conda env create --file environment.yml
conda activate SparseGS
```
We suggest using **CUDA 12** but CUDA 11 should work. You may need to change the cudatoolkit and pytorch version in the .yml file. 

## Data Preparation
For data preparation, we use the same COLMAP preprocessing pipeline as the original 3DGS. If you are not familiar with the format below, use convert.py as reference. Your input data folder should look like this:
``` 
data -- scene -- images -- image1.JPG
              |         |
              |         -- image2.JPG
              |         |
              |         -- ......
              |
              |
              -- sparse -- 0 -- cameras.bin
                             |
                             -- images.bin
                             |
                             -- points3D.bin
``` 
When dealing with extreme sparse cases, COLMAP will fail to reconstruct a pointcloud or estimate poses. We provide a heuristic to extract COLMAP model for sparse scenes from their dense models. Please see the scripts in [dataprep_scripts](dataprep_scripts).

We also provide some preprocessed data with various numbers of input view. You may download them [here](https://drive.google.com/drive/folders/1VVkkPw_ubQ0A3052ErDK4PyDgT9hqfKJ?usp=sharing). Please cite our project if you find these data useful.

## Getting Monocular-estimated Depth Maps
You can use any monocular depth estimation model to generate your gt depth maps as long as they are saved as .npy files. In this project, we have integrated "S. Mahdi H. Miangoleh, BoostingMonocularDepth, (CVPR 2022)". If you would like to use this, please download the model checkpoints **latest_net_G.pth**, **res101.pth** from [here](https://drive.google.com/drive/folders/1VVkkPw_ubQ0A3052ErDK4PyDgT9hqfKJ?usp=sharing) and put them in [BoostingMonocularDepth/pix2pix/checkpoints/mergemodel](BoostingMonocularDepth/pix2pix/checkpoints/mergemodel). Then, run:

```
python3 prepare_gt_depth.py <path-to-image-folder> <path-to-depth-save-folder>

e.g. python3 prepare_gt_depth.py "./data/kitchen_12/images" "./data/kitchen_12/depths"
```
After preping the depth maps, your input folder should look like:
```
e.g.
data -- kitchen_12 -- images -- DSCF0656.JPG
                  |          |
                  |          -- DSCF0657.JPG
                  |          |
                  |          -- ......
                  |
                  |
                  -- depths -- DSCF0656.npy
                  |         |
                  |         -- DSCF0657.npy
                  |         |
                  |         -- ......
                  |
                  |
                  -- sparse -- 0 -- cameras.bin
                                |
                                -- images.bin
                                |
                                -- points3D.bin
```



## Training
Train SparseGS:
``` 
python3 train.py --source_path data/kitchen_12 --model_path output/kitchen_12_SparseGS --beta 5.0 --lambda_pearson 0.05 --lambda_local_pearson 0.15 --box_p 128 --p_corr 0.5 --lambda_diffusion 0.001 --SDS_freq 0.1 --step_ratio 0.99 --lambda_reg 0.1 --prune_sched 20000 --prune_perc 0.98 --prune_exp 7.5 --iterations 30000 --checkpoint_iterations 30000 -r 2
``` 
There are a lot of arguments here, below is a cheat sheet. When lambda_* is set to 0, the corresponding component is disabled. If prun_sched is not set, floater pruning is disabled.
```
    # ================ Pearson Depth Loss =============
    self.lambda_pearson = 0.0 # weight for global pearson loss

    self.lambda_local_pearson = 0.0 # weight for patch-based pearson loss
    self.box_p = 128 # patch size
    self.p_corr = 0.5 # number of patches sampled at each iteration

    self.beta = 5.0 # Temperature value for the softmax depth function
    # ==================================================

    # ================= Floater Pruning ==============
    self.prune_sched = 20000 # Floater Pruning iteration 
    self.prune_exp = 7.0 # lower is less aggresive
    self.prune_perc = 0.98 # higher is less aggresive
    # ================================================

    # ======== Diffusion(SDS Loss) Params ============
    # Note: SDS Loss kicks in after two-thirds of the training iterations

    self.step_ratio = 0.95 # lower is more noisy, = 1 means no diffusion
    self.lambda_diffusion = 0.0
    self.SDS_freq = 0.1 # SDS apply frequency, = 1 means every iteration
    # ================================================

    # =========== Depth Warping Loss Params ===========
    self.lambda_reg = 0.0
    self.warp_reg_start_itr = 4999 # warping loss kicks in iteration
    # ================================================
```


## Rendering
Run the following script to render the images. The render images will be saved under **renders** inside your model folder. Note that the **--source_path** needs to point to the full data folder (not the sparse one used for training). You may download the full data [here](https://drive.google.com/drive/folders/1VVkkPw_ubQ0A3052ErDK4PyDgT9hqfKJ?usp=sharing).

```
python3 render.py --source_path ./data/kitchen --model_path ./output/kitchen_12_SparseGS --no_load_depth --iteration 30000
```


## Evaluation
Please run the following script to evaluate your model. **--exclude_path** should point to your training data since they should be excluded when calculating metrics.

```
python3 metrics.py --model_path ./output/kitchen_12_SparseGS --exclude_path ./data/kitchen_12
```

## Acknowledgement

Special thanks to the following awesome projects!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [FSGS](https://github.com/VITA-Group/FSGS)
- [BoostingMonocularDepth](https://github.com/compphoto/BoostingMonocularDepth?tab=readme-ov-file)
- [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)

## Citation
If you find this project interesting, please consider citing our paper.

```
@misc{xiong2023sparsegs,
author = {Xiong, Haolin and Muttukuru, Sairisheek and Upadhyay, Rishi and Chari, Pradyumna and Kadambi, Achuta},
title = {SparseGS: Real-Time 360° Sparse View Synthesis using Gaussian Splatting},
journal = {Arxiv},
year = {2023},
}
```
