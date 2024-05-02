from BoostingMonocularDepth.prepare_depth import prepare_gt_depth 
import sys


# input_image_folder = './data/kitchen_12/images'
# output_depth_folder = './data/kitchen_12/depths'
input_image_folder = sys.argv[1] 
output_depth_folder = sys.argv[2]
prepare_gt_depth(input_folder = input_image_folder, save_folder = output_depth_folder)