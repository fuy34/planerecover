# Plane-Recover

This codebase is a TensorFlow implementation of our ECCV-2018 paper:

[Recovering 3D Planes from a Single Image via Convolutional Neural Networks](https://faculty.ist.psu.edu/zzhou/paper/ECCV18-plane.pdf)

[Fengting Yang](http://personal.psu.edu/fuy34/), [Zihan Zhou](https://faculty.ist.psu.edu/zzhou/Home.html)

Please contact Fengting Yang (fuy34@psu.edu) if you have any question.

## Prerequisites
This codebase was developed and tested with python 2.7, Tensorflow 1.4.1, CUDA 8.0.61 and Ubuntu 16.04.

## Preparing training data
[Here](https://psu.box.com/s/6ds04a85xqf3ud3uljjxnedmux169ebf) we provide our training and testing data on [SYNTHIA](http://synthia-dataset.net/) dataset. Once you download the training data, you can set the training data path as <SYNTHIA_DUMP_DIR> in training command and start to train the network. 

If you wish to generate the training data by yourself, you may want to follow the following steps.

First, download the four-season sequences  (Spring, Summer, Fall, Winter) of SEQS-02, SEQS-04, SEQS-05, and save them in one folder ```<SYNTHIA_DIR>```. Then run the following command to filter out the static frames and generate the training data
```
python data_pre_processing/SYNTHIA/SYNTHIA_frame_filter.py --dataset_dir=<SYNTHIA_DIR> --dump_root=<SYNTHIA_DUMP_Filtered_DIR> 
python data_pre_processing/SYNTHIA/SYNTHIA_pre_processing.py --filtered_dataset=<SYNTHIA_DUMP_Filter_DIR> --dump_root=<SYNTHIA_DUMP_DIR> 
```
The code will generate two "*.txt*" files for training and testing, we recommend to replace the ```tst_100.txt``` with the one in the ```data_pre_processing/SYNTHIA``` folder for the availablity of the ground truth. The "train_8000.txt" in the some folder records the training data we used in our training. Please note the depth unit of SYNTHIA is centimeter, so we divide the depth map by 100.0 in data loading process.  


## Training
Once the data is prepared, you should be able to train the model by running the following command
```
python train.py --dataset_dir=<SYNTHIA_DUMP_DIR> --log_dir=<CKPT_LOG_DIR>
```

if you want to continue to train or fine-tune from a pre-trained model, you can run 
```
python train.py --dataset_dir=<SYNTHIA_DUMP_DIR> --log_dir=<CKPT_LOG_DIR> --init_checkpoint_file=<PATH_TO_THE_CKPT> --continue_train=True
```

You can then start a `tensorboard` session by
```
tensorboard --logdir=<DIR_CONTAINS_THE_EVENT_FILE> --port=6006
```
and monitor the training progress by opening the 6006 port on your browser. If everything is set up properly, reasonable segmenation should be observed around 200k steps. The number of recovered planes will keep increase until it reaches the maximum number set in the code (default=5). 

A pre-trained model has been included in the folder named "pre_trained_model", and the ground truth segmentation is in "eval/labels/".
 
## Testing
We provide test code to generate: 1) plane segmentation (and its visualization) and 2) depth prediction (planar area only). The evaluation of the depth prediction accuracy will be presented right after the test process. Please run
```
python test_SYNTHIA.py --dataset=<SYNTHIA_DUMP_Filtered_DIR> --output_dir=<TEST_OUTPUT_DIR> --test_list=<Tst_100.txt in SYNTHIA_DUMP_DIR> --ckpt_file=<TRAINED_MODEL>
```
Note: 
1. We use the ```filtered data``` as input instead of the ```pre-processed``` one (to preserve the resolution of the ground truth depth). If you do not want to do the pre-processing and already download our data, you can simply modify the path related to the dataset in ```test_SYNTHIA.py```. The final result may not be exactly the same as ours, but should be similar.
2. We intentionally exinclude seq.22 in our training to test the model performance in a video sequence. That is why this sequence is missing in the provided training/test data. The ```filtered seq.22``` (without pre-processing) can be download [here](https://psu.box.com/s/9rpxfa8zasy95ia5u0ol0wxm6qj9i7s8).
3. The code to generate planar 3D model is updated ```eval/generate3D.py```. It should works once all the hard-code path is set correctly according to your local environment. The output will be ```.ply``` file, which can be visualized in MeshLab directly. 

## Evaluation
We also provide the MATLAB code for evaluation of plane segmentation accuracy:

(1) Open the ```eval/eval_planes.m```;  
(2) Set the ```pred_path``` as the path to the ```plane_sgmts``` folder generated in test step and check if the ```label_path``` is appropriately pointing to the ```eval/labels/SYN_GT_sgmt```;  
(3) Run the program, you should be able to see the evaluation result on the command window.  

## Acknowledgement
Our code is developed based on the training framework provided by [SfMLearner](https://github.com/tinghuiz/SfMLearner)


