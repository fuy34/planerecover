from __future__ import division
import numpy as np
from glob import glob
import os
import cv2

# support code for SYNTHIA_pre_processing.py

class SYNTHIA_data_loader(object):
    def __init__(self,
                 dataset_dir,
                 img_height=192,
                 img_width=320):
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.train_seqs = [  0, 1, 2, 3, \
                            4, 5, 6, 7, \
                            8, 9, 10, 11, \
                            12, 14, 13, 15, \
                            17, 19, 16, 18, \
                            20, 21,  23,\
                           24, 25, 26, 27,\
                           28, 29, 30, 31,\

                           32, 33, 34, 35,\
                           36,  38, 37, 39,\
                           40, 41, 42, 43,\
                           44, 45, 46, 47,\
                           ] #NOTE: the depth image 000592 in seq.22 cannot be read by cv2. so seq.22 is excluded


        self.collect_train_frames()


    def collect_train_frames(self):
        self.train_frames = []
        for seq in self.train_seqs:
            seq_dir = os.path.join(self.dataset_dir, '%.2d' % seq,'images')
            seq_list = (glob(seq_dir + '/*.png') ) #glob is used for searching the files comforms to the path rules
            seq_list.sort(key=lambda f: int(filter(str.isdigit, f)))
            # name_seq_list = os.path.basename(seq_list)
            for n in seq_list:
                name_seq,_ = os.path.basename(n).split(".")
                self.train_frames.append('%.2d ' %seq + name_seq) #TODO: here I assume the frames should be corrponsponding
        self.num_train = len(self.train_frames)


    def load_image_sequence(self, frames, tgt_idx):
        image_seq = []
        curr_drive, curr_frame_id = frames[tgt_idx].split(' ')
        curr_img = self.load_image(curr_drive, curr_frame_id)
        curr_depth = self.load_depth_sequence(curr_drive, curr_frame_id)
        curr_depth = self.resize_near(curr_depth, self.img_width, self.img_height)
        curr_label = self.load_label_sequence(curr_drive, curr_frame_id)
        curr_label = self.resize_near(curr_label, self.img_width, self.img_height)
        zoom_y = float(self.img_height) / curr_img.shape[0]
        zoom_x = float(self.img_width) / curr_img.shape[1]
        curr_img = self.resize_near(curr_img,self.img_width, self.img_height ) #scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
        image_seq.append(curr_img)
        return image_seq,curr_depth, curr_label, zoom_x, zoom_y

    def load_example(self, frames, tgt_idx):
        image_seq, depth, label, zoom_x, zoom_y = self.load_image_sequence(frames, tgt_idx)
        tgt_drive, tgt_frame_id = frames[tgt_idx].split(' ')  #tgt_drive : seq.num, tgt_frame_id: the frame num
        intrinsics = self.load_intrinsics(tgt_drive, tgt_frame_id)
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)

        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = tgt_drive
        example['file_name'] = tgt_frame_id
        example['depth'] = depth
        example['label'] = label

        return example

    def get_train_example_with_idx(self, tgt_idx):
        example = self.load_example(self.train_frames, tgt_idx)
        return example

    def load_label_sequence(self, frames, tgt_idx):
        labbel_file = os.path.join(self.dataset_dir, '%s/label/%s.png' % (frames, tgt_idx))
        label = cv2.imread(labbel_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return label

    def load_depth_sequence(self,frames,tgt_idx):
        depth_file = os.path.join(self.dataset_dir, '%s/depth/%s.png' % (frames, tgt_idx))
        depth =  cv2.imread(depth_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return depth

    #TODO:try to only use this once for each seq
    def load_valid_list(self,tgt_drive):
        txt_name = "/valid_frms.txt"
        s_path = self.dataset_dir + tgt_drive + txt_name
        lst = np.loadtxt(s_path, dtype='int')
        return lst

    def load_image(self, drive, frame_id):
        img_file = os.path.join(self.dataset_dir, '%s/images/%s.png' % (drive, frame_id))
        img =  cv2.imread(img_file, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        return img

    def load_intrinsics(self, drive, frame_id):
        calib_file = os.path.join(self.dataset_dir, '%s/intrinsics.txt' % drive)
        # tmp = np.genfromtxt(calib_file)
        with open(calib_file,'r') as f:
            C = f.readlines()

        f_x=  np.genfromtxt([C[0]]) #C[0].split("\n")[0]
        f_y= f_x
        c_x =  np.genfromtxt([C[2]]) #C[2].split("\n")
        c_y =  np.genfromtxt([C[4]]) #C[4].split("\n")
        intrinsics = np.array([f_x,  c_x,  f_y,c_y]) #np.genfromtxt([f_x, c_x, f_y, c_y])

        return intrinsics


    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0] *=sx
        out[1] *=sx
        out[2] *=sy
        out[3] *=sy

        return out

    # the one could be replaced by cv2.resize with cv2.INTER_NEARST
    def resize_near(self, img, dst_w, dst_h):
        h = img.shape[0]
        w = img.shape[1]
        zoom_x_up = (float(w) / dst_w)
        zoom_y_up = (float(h) / dst_h)
        pixel_coords = self.meshgrid(dst_h, dst_w)
        x = (pixel_coords[0,:,:] * zoom_x_up).astype('int')
        y = (pixel_coords[1, :, :] * zoom_y_up).astype('int')

        if img.ndim == 3:
            dst_img = img[y, x, :]
        elif img.ndim == 2:
            dst_img = img[y,x]

        return dst_img

    def meshgrid(self, height, width, is_homogeneous=True):
        """Construct a 2D meshgrid.

        Args:
          batch: batch size
          height: height of the grid
          width: width of the grid
          is_homogeneous: whether to return in homogeneous coordinates
        Returns:
          x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
        """
        x_t = np.matmul(np.ones(shape=[height, 1]), np.expand_dims(np.linspace(-1., 1, width), 1).T)
        y_t = np.matmul(np.expand_dims(np.linspace(-1.0, 1.0, height), 1), np.ones(shape=np.stack([1, width])))

        x_t = (x_t + 1.0) * 0.5 * (width - 1)
        y_t = (y_t + 1.0) * 0.5 * (height - 1)
        if is_homogeneous:
            ones = np.ones_like(x_t)
            coords = np.stack([x_t, y_t, ones], axis=0)
        else:
            coords = np.stack([x_t, y_t], axis=0)
        return coords
