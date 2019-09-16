import os
import numpy as np
import scipy.misc
import PIL.Image as pil
import cv2

'''
Generate 3D model from the input depth
@author -- Fengting Yang 
@created time -- Mar.10 2018

@Usage:
   1. Set TEST_LIST and all the DIR. Note for dir, a final '/' is required
   2. Ensure a right intrinsics, resize it if resizing img to a new size
    

@parameters:
   1. DATASET_DIR:      The path of dataset (for SYN, use the filtered images)
   2. PRED_DEPTH_DIR:   The path of predicted depth 
   3. OUTPUT_DIR:       The path to save the output model 
   4. intrinsics:       The camera intrinsics, note if resizing the image, the please resize the intrinsics as well
   5. scalingFactor:    The factor of depth. e.g. for SYNTHIA, the depth is stored as "cm", so to "m", it should be 100.
   6. MAX_DEPTH:        The upper boundary of the depth that will be used to generate the 3D model  
   7. MIN_DEPTH:        The lower boundary of the depth that will be used to generate the 3D model
   8. TEST_LIST:        The txt list of the test image in a format "xx yyyyyy", xx is the seq_id, yyyyyy is the img_id
'''

# 320*192
# intrinsics =  np.array(([[133.185088,0.,160.000000], [ 0., 134.587036,96.000000], [0., 0., 1.]]))
# focalLength_x =133.185075
# focalLength_y = 134.587036
# centerX = 160.000000
# centerY =  96.000000
# scalingFactor = 100. #depth scale
# IMG_WIDTH = 320
# IMG_HEIGHT = 192


# 640*380
intrinsics =  np.array(([[266.370176,0.,320.000000], [ 0., 266.370176,190.000000], [0., 0., 1.]]))
focalLength_x =266.370176
focalLength_y =  266.370176
centerX = 320.000000
centerY =  190.000000
scalingFactor = 100. #depth scal
IMG_WIDTH = 640
IMG_HEIGHT = 380


MAX_DEPTH = 100. #depth scale
MIN_DEPTH = 0.1

DATASET_DIR = '/data/Filtered_SYN_data_02_04_05_four_seasons_fwd_bwd/'
PRED_DEPTH_DIR = '/data/depth_training_res/depth_tst_res/SYN_FB_all_new_size_lR=0.0001_min_depth=0.1_max_depth=100.0_depth_moved_mask_l2_smooth_loss_BN_test_on_selected/'
OUTPUT_DIR = '/data/plane_detection/SYN_FB_all_lR=0.0001_min_depth=0.1_max_depth=100.0_depth_moved_mask_l2_smooth_loss_BN_test_on_selected_half_size/'
TEST_LIST = '/home/fuy34/Plane_detection/SYN_FB_selected_new_size/tst_100.txt'

def generate_pointcloud(rgb, depth, ply_file):
    points = []
    for u in range(rgb.shape[1]):
        for v in range(rgb.shape[0]):
            color = rgb[v,u,:]
            Z = depth[v,u] / scalingFactor
            if Z <= MIN_DEPTH or Z>= MAX_DEPTH :
                continue
            X = float(u - centerX) * Z / focalLength_x
            Y = float(v - centerY) * Z / focalLength_y
            points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
    file = open(ply_file, "w")
    file.write('''ply
    format ascii 1.0
    element vertex %d
    property float x
    property float y
    property float z
    property uchar red
    property uchar green
    property uchar blue
    property uchar alpha
    end_header
    %s
    ''' % (len(points), "".join(points)))
    file.close()

def main():
  with open(TEST_LIST, 'r') as f:  #the test list
        test_files_list = []
        depth_file_list = []
        test_files = f.readlines()
        for t in test_files:
            t_split = t[:-1].split()
            test_files_list.append(DATASET_DIR + t_split[0] +'/images/'+ t_split[-1] + '.png' )
            depth_file_list.append(PRED_DEPTH_DIR + t_split[0] + '_' + t_split[-1] + '_pred.png')

  if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

  for idx in range(len(test_files_list)):
        img = pil.open(test_files_list[idx])
        img = np.array(img)
        depth = cv2.imread(depth_file_list[idx], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

        img_resize = cv2.resize(img,(IMG_WIDTH, IMG_HEIGHT),cv2.INTER_NEAREST) #,cv2.INTER_AREA
        depth_resize = cv2.resize(depth, (IMG_WIDTH, IMG_HEIGHT),cv2.INTER_NEAREST) #, cv2.INTER_AREA

        name = test_files_list[idx].split('/')
        model_path = OUTPUT_DIR
        ply_name = name[-3] + '_' + name[-1][:-4] + '.ply'
        generate_pointcloud(img_resize, depth_resize, model_path + ply_name)

        # To save the corresponding depth and rgb image (optional)
        # depth_norm = np.copy(depth)
        # cv2.normalize(depth, depth_norm, 65535, 0, cv2.NORM_MINMAX);
        #
        # depth_name = name[-3] + '_' + name[-1][:-4] + '_depth.png'
        # cv2.imwrite( model_path + depth_name, depth_norm)
        #
        # png_name = name[-3] + '_' + name[-1][:-4] + '.png'
        # scipy.misc.imsave(model_path + png_name, img)




if __name__ == '__main__':
    main()
