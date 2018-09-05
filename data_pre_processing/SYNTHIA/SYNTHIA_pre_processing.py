from __future__ import division
import argparse
import numpy as np
from glob import glob
from joblib import Parallel, delayed
import os
import cv2

import random
'''
SYNTHIA_data preProcessor 

@author -- Fengting Yang 
@created time -- Feb.01. 2018

@Usage:
    This file is designed to work with SYNTHIA_data_loader for Planet/SfMLearner-like pre-processing
    
    To use it:
    1. ensure all the data is filterd by SYNTHIA_frame_filter.py, 
    2. ensure all the poses and image files are renamed by SYNTHIA_frame_rename.py
    3. locate the well organized SYNTHIA seqs and choose a dump_root to save the pre-processed data   

'''

parser = argparse.ArgumentParser()
parser.add_argument("--filtered_dataset", type=str, default="", help="where the filtered dataset is stored",required=True)
parser.add_argument("--dump_root", type=str, default="", help="Where to dump the data",required=True)
parser.add_argument("--img_height", type=int, default=192, help="image height") 
parser.add_argument("--img_width", type=int, default=320, help="image width")  
parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
args = parser.parse_args()



def concat_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res

def dump_example(n):
    if n % 100 == 0:
        print('Progress %d/%d....' % (n, data_loader.num_train))
    example = data_loader.get_train_example_with_idx(n)
    if example == False:
        return
    image_seq = concat_image_seq(example['image_seq'])
    depth = example['depth']
    label = example['label']
    intrinsics = example['intrinsics']
    dump_dir = os.path.join(args.dump_root, example['folder_name'])
    if not os.path.isdir(dump_dir):
    #     os.makedirs(dump_dir, exist_ok=True)
        try:
            os.makedirs(dump_dir)
        except OSError:
            if not os.path.isdir(dump_dir):
                raise

    # save images
    dump_img_file = dump_dir + '/%s.jpg' % example['file_name']
    cv2.imwrite(dump_img_file, image_seq.astype(np.uint8))
    #
    # #save depth
    dump_depth_file = dump_dir + '/%s_depth.png' % example['file_name']
    cv2.imwrite(dump_depth_file, depth.astype(np.uint16))  

    # save label
    dump_label_file = dump_dir + '/%s_label.png' % example['file_name']
    cv2.imwrite(dump_label_file, label.astype(np.uint8))

    #save intrinsic
    dump_cam_file = dump_dir + '/%s_cam.txt' % example['file_name']
    with open(dump_cam_file, 'w') as f:
        f.write('%f,0.,%f,0.,%f,%f,0.,0.,1.' % (intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]))
    dump_pose_file = dump_dir + '/%s_pose.txt' % example['file_name']



def main():
    dump_root = args.dump_root
    if not os.path.exists(dump_root):
        os.makedirs(dump_root)

    global data_loader
    from SYNTHIA_data_loader import SYNTHIA_data_loader
    data_loader = SYNTHIA_data_loader(args.filtered_dataset,
                                        img_height=args.img_height,
                                        img_width=args.img_width)

    # the following code is for debug
    for n in range(data_loader.num_train):
        dump_example(n)

    # mutil-thread running for speed
    # Parallel(n_jobs=args.num_threads)(delayed(dump_example)(n) for n in range(data_loader.num_train))

    # Split into train/val
    np.random.seed(1)
    subfolders = os.listdir(dump_root)
    with open(dump_root + 'train0.txt', 'w') as tf:
        with open(dump_root + 'test0.txt', 'w') as vf:
            for s in subfolders:
                if not os.path.isdir(dump_root + '/%s' % s):
                    continue
                imfiles = glob(os.path.join(dump_root, s, '*.jpg'))
                frame_ids = [os.path.basename(fi).split('.')[0] for fi in imfiles]
                for frame in frame_ids:
                    if np.random.random() < 0.1:
                        vf.write('%s %s\n' % (s, frame))
                    else:
                        tf.write('%s %s\n' % (s, frame))

    train_num = 8000
    test_num = 100
    TRAIN_LIST = dump_root + 'train0.txt'
    TEST_LIST = dump_root + 'test0.txt'
    with open(TRAIN_LIST, 'r') as f:
        train_list = f.readlines()
        train_choices = random.sample(train_list, train_num)

        dump_root = os.path.split(TRAIN_LIST)[0]
        with open(dump_root + '/train_' + str(train_num) + '.txt', 'w') as tf:
            for line in train_choices:
                tf.write(line)
            print("train list selected")

    with open(TEST_LIST, 'r') as f:
        tst_list = f.readlines()
        tst_choices = random.sample(tst_list, test_num)

        dump_root = os.path.split(TEST_LIST)[0]
        with open(dump_root + '/tst_' + str(test_num) + '.txt', 'w') as vf:
            for line in tst_choices:
                vf.write(line)
            print("test list selected")


if __name__ == '__main__':
    main()

