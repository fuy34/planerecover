import os
import shutil
import numpy as np
import cv2
import argparse
'''
Frame filter for SYNTHIA dataset
@author -- Fengting Yang 
@created time -- Dec.21 2017

@Usage:
   1. type the seq name we want in str_Seq_list
   2. set the root path of these seqs in str_Seq_root_path 
   3. set the str_dump_path as the path we want the filtered seqs to be stored 
   4. the qualified seqs will be saved in the dump_path directly

@parameters:
    thres:  the threshold of the lowest accpetable speed
    log:    the txt store the removed frame names 
    
@Note: 
    we only use forward and backward frames in our training 
'''

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default="", help="where the downloaded dataset is",required=True)
parser.add_argument("--dump_root", type=str, default="",help="Where to dump the filtered data",required=True)
args = parser.parse_args()

#TODO: rewirte the following variables when necessary
#***********************************************************
n_seq = 12
n_camera = 2   # Omni_F and Omni_B
n_stereo =2    # Stereo_Left and Stereo_Right

str_Seq_list      = ['SYNTHIA-SEQS-02-SPRING', 'SYNTHIA-SEQS-02-SUMMER','SYNTHIA-SEQS-02-FALL', 'SYNTHIA-SEQS-02-WINTER',
                     'SYNTHIA-SEQS-04-SPRING', 'SYNTHIA-SEQS-04-SUMMER', 'SYNTHIA-SEQS-04-FALL','SYNTHIA-SEQS-04-WINTER',
                     'SYNTHIA-SEQS-05-SPRING', 'SYNTHIA-SEQS-05-SUMMER', 'SYNTHIA-SEQS-05-FALL','SYNTHIA-SEQS-05-WINTER',]

str_Seq_root_path = args.dataset_dir
str_log_path      = "/copy_log.txt"
str_dump_path     = args.dump_root

thres = 0.8     #speed threshold

#*******************************************************
# fixed format for SYNTHIA
str_pose_list   = ['/CameraParams/Stereo_Left/','/CameraParams/Stereo_Right/']
str_img_list    = ['/RGB/Stereo_Left/','/RGB/Stereo_Right/']
str_depth_list  = ['/Depth/Stereo_Left/','/Depth/Stereo_Right/']
str_label_list  = ['/GT/LABELS/Stereo_Left/','/GT/LABELS/Stereo_Right/']
str_camera_list = ['Omni_F/','Omni_B/']

intrinsic_path = os.path.join(args.dataset_dir,'SYNTHIA-SEQS-02-SPRING/CameraParams/intrinsics.txt')
#*******************************************************

for i in range(n_seq):
    for j in range(n_stereo):
        for k in range (n_camera):
                load_img_path = str_Seq_root_path + str_Seq_list[i] + str_img_list[j] + str_camera_list[k]
                load_depth_path = str_Seq_root_path + str_Seq_list[i] + str_depth_list[j] + str_camera_list[k]
                load_label_path = str_Seq_root_path + str_Seq_list[i] + str_label_list[j] + str_camera_list[k]
                load_pose_path = str_Seq_root_path + str_Seq_list[i] + str_pose_list[j] + str_camera_list[k]

                pose_files = os.listdir(load_pose_path)                            # the files must be sorted, as it is disordered initially
                pose_files.sort(key=lambda f: int(filter(str.isdigit, f)))

                cnt = 0         #counter for the qualified files
                qualified_imgs = []
                qualified_poses = []
                pre_t = np.array([0., 0., 0.])
                total_cnt = len(pose_files)

                for pose_file in pose_files:
                    if not os.path.isdir(pose_file):
                        R_t = np.genfromtxt(load_pose_path + pose_file)
                        t = R_t[-4:-1]
                        if np.linalg.norm((t - pre_t)) > thres:
                            name = pose_file.split('.')[0]
                            cnt += 1
                            qualified_imgs.append(name + '.png')
                            qualified_depth = qualified_imgs
                            qualified_label = qualified_imgs
                            qualified_poses.append(name + '.txt')
                            pre_t = t

                cpy_path = str_dump_path + "%.2d"%(i * n_stereo * n_camera+ j * n_camera + k)

                for n in range(cnt):
                    # copy frames
                    if os.path.isfile(load_img_path + qualified_imgs[n]):
                        if not os.path.isdir(cpy_path + "/images"):
                            os.makedirs(cpy_path + "/images")
                        shutil.copy(load_img_path + qualified_imgs[n],cpy_path + "/images")
                    else:
                        print("Wrong image path")

                    # copy depth
                    if os.path.isfile(load_depth_path + qualified_depth[n]):
                        if not os.path.isdir(cpy_path + "/depth"):
                            os.makedirs(cpy_path + "/depth")
                        shutil.copy(load_depth_path + qualified_depth[n],cpy_path + "/depth")
                    else:
                        print("Wrong image path")

                    # copy label
                    if os.path.isfile(load_label_path + qualified_label[n]):
                        if not os.path.isdir(cpy_path + "/label"):
                            os.makedirs(cpy_path + "/label")
                        seg = cv2.imread(load_label_path + qualified_label[n], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

                        #***********************pre-processed the labels to get potential plane******************
                        seg_id = seg[:, :, 2]
                        dump = np.zeros(shape=seg_id.shape)
                        dump[seg_id >= 2] = 1
                        dump[seg_id > 5] = 0
                        dump[seg_id == 12] = 1 # the id = 2 3 4 5 12 are potential planes
                        #*******************note the code above if the original label is wanted*******************

                        write_path = cpy_path + "/label/" + qualified_depth[n]
                        cv2.imwrite(cpy_path + "/label/" + qualified_depth[n], dump)
                    else:
                        print("Wrong image path")

                # copy intrinsic -- for every seq in synthia, the instrinsics are same
                shutil.copy(intrinsic_path, cpy_path)
                print("%d files have been copied from " %cnt + load_img_path + "to" + cpy_path)

                # write log files
                with open((cpy_path + str_log_path), "w+") as log:
                    print >> log, ("copied files \n" + str(qualified_imgs) + "\n copied %d files from totally %d files" %(cnt,total_cnt))


