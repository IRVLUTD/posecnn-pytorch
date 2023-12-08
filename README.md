# posecnn-pytorch

PyTorch implementation of the PoseCNN and PoseRBPF framework.

### License

PoseCNN-PyTorch is released under the NVIDIA Source Code License (refer to the LICENSE file for details).

### Citation

If you find the package is useful in your research, please consider citing:

    @inproceedings{xiang2018posecnn,
        Author = {Xiang, Yu and Schmidt, Tanner and Narayanan, Venkatraman and Fox, Dieter},
        Title = {PoseCNN: A Convolutional Neural Network for 6D Object Pose Estimation in Cluttered Scenes},
        booktitle   = {Robotics: Science and Systems (RSS)},
        Year = {2018}
    }

    @inproceedings{deng2019pose,
        author    = {Xinke Deng and Arsalan Mousavian and Yu Xiang and Fei Xia and Timothy Bretl and Dieter Fox},
        title     = {PoseRBPF: A Rao-Blackwellized Particle Filter for 6D Object Pose Tracking},
        booktitle = {Robotics: Science and Systems (RSS)},
        year      = {2019}
    }

### Required environment

- Ubuntu 16.04 or above
- PyTorch 0.4.1 or above
- CUDA 9.1 or above

### Installation

Use python3 with Conda.

0. Create a conda environment
   ```Shell
   conda create -n posecnn
   ```

1. Install [PyTorch](https://pytorch.org/)

2. Install Eigen from the Github source code [here](https://github.com/eigenteam/eigen-git-mirror)

3. Install python packages
   ```Shell
   pip install -r requirement.txt
   ```

4. Initialize the submodules
   ```Shell
   git submodule update --init --recursive
   ```
   
5. Install Sophus under the root folder

6. Compile the new layers under $ROOT/lib/layers we introduce in PoseCNN
    ```Shell
    cd $ROOT/lib/layers
    sudo python setup.py install
    ```

7. Compile cython components
    ```Shell
    cd $ROOT/lib/utils
    python setup.py build_ext --inplace
    ```

8. Compile the ycb_render in $ROOT/ycb_render
    ```Shell
    cd $ROOT/ycb_render
    sudo python setup.py develop
    ```
    
9. Install ROS in conda
    ```Shell
    conda install -c conda-forge rospkg empy
    ```

### Download

- 3D models of YCB Objects we used [here](https://utdallas.box.com/s/ls01mxdyai1ui6lflejk1n8x86ve20oy) (3G). Save under $ROOT/data or use a symbol link.

- PoseCNN checkpoints from [here](https://utdallas.box.com/s/g0qnbs615kcizcqvys6pn96dr251m0lk)

- PoseRBPF checkpoints from [here](https://utdallas.box.com/s/k5phfynz6evjln9nodc7fqo13veoy4aq)

- Our real-world images with pose annotations for 20 YCB objects collected via robot interation [here](https://drive.google.com/file/d/1cQH_dnDzyrI0MWNx8st4lht_q0F6cUrE/view?usp=sharing) (53G). Check our ICRA 2020 [paper](https://arxiv.org/abs/1909.10159) for details.


### Training and testing on the YCB-Video dataset
1. Download the YCB-Video dataset from [here](https://rse-lab.cs.washington.edu/projects/posecnn/).

2. Create a symlink for the YCB-Video dataset
    ```Shell
    cd $ROOT/data/YCB_Video
    ln -s $ycb_data data
    ```

3. Training and testing on the YCB-Video dataset
    ```Shell
    cd $ROOT

    # multi-gpu training, use 1 GPU or 2 GPUs since batch size is set to 2
    ./experiments/scripts/ycb_video_train.sh

    # testing, $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ycb_video_test.sh $GPU_ID

    ```
    
### Realsense camera set up
1. Install realsense ros package
    ```Shell
    sudo apt install ros-noetic-realsense2-camera
    ```
    
2. Install realsense SDK from [here](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md)

### Running on Realsense cameras or Fetch

1. Run PoseCNN detection for YCB objects
    ```Shell
    # run posecnn for detection (20 YCB objects), $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ros_ycb_object_test_subset_poserbpf_realsense_ycb.sh $GPU_ID $INSTANCE_ID
    ```

2. Run PoseBPRF for YCB objects
    ```Shell
    # $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ros_poserbpf_ycb_object_test_subset_realsense_ycb.sh $GPU_ID $INSTANCE_ID
    
<p align="center"><img src="data/example.png" width="640" height="320"/></p>    

**Changing the set of objects to be used for pose detection:**
- See the object ordering in `lib/datasets/ycb_object.py` defined under `YCBObject.__init__()` under the var name `self._classes_all`
- The order in which object names appear is used in the config file while determining which classes to track and do pose estimation for.
  - Note the python list/tuple is *zero-indexed*
- Config file: `experiments/cfgs/ycb_object_subset_realsense.yml`
  - The objects to be used are specified under `TEST.CLASSES` which is tuple with indices corresponding to the above list `self._classes_all`
  - Example: If your object set is `003, 005, 007`, then the tuple specififed in yaml file will be: `[2, 4, 6]`.
  - You can keep `TEST.SYMMETRY` to be a list with all zeros and matching the length of `TEST.CLASSES` in the yaml file.

### Running ROS Kitchen System with YCB Objects
1. Start Kinect for tracking kitchen
    ```Shell
    roslaunch lula_dart multi_kinect.launch
    ```

2. Start DART
    ```Shell
    roslaunch lula_dart kitchen_dart_kinect2.launch
    ```

3. Run DART stitcher
    ```Shell
    ./ros/dart_stitcher_kinect2.py 
    ```

4. Start realsense
    ```Shell
    roslaunch realsense2_camera rs_aligned_depth.launch tf_prefix:=measured/camera
    ```

5. Run PoseCNN detection for YCB objects
    ```Shell
    # run posecnn for detection (20 YCB objects and cabinet handle), $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ros_ycb_object_test_subset_poserbpf_realsense_ycb.sh $GPU_ID $INSTANCE_ID
    ```

6. Run PoseBPRF for YCB objects
    ```Shell
    # $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ros_poserbpf_ycb_object_test_subset_realsense_ycb.sh $GPU_ID $INSTANCE_ID
    ```

7. (optional) Run PoseCNN detection for blocks
    ```Shell
    # run posecnn for detecting blocks, $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ros_ycb_object_test_subset_poserbpf_realsense.sh $GPU_ID $INSTANCE_ID
    ```

8. (optional) Run PoseBPRF for blocks
    ```Shell
    # $GPU_ID can be 0, 1, etc.
    ./experiments/scripts/ros_poserbpf_ycb_object_test_subset_realsense.sh $GPU_ID $INSTANCE_ID
    ```
