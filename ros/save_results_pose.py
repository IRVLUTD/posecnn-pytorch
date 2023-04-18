#!/usr/bin/env python

# --------------------------------------------------------
# PoseCNN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""collect images from Fetch"""

import rospy
import message_filters
import cv2
import argparse
import threading
import pprint
import time, os, sys
import os.path as osp
import numpy as np
import tf
import tf2_ros
import tf.transformations as tra
import scipy.io
from transforms3d.quaternions import mat2quat, quat2mat
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from posecnn_pytorch.msg import DetectionList, Detection, BBox
from sensor_msgs.msg import CameraInfo

lock = threading.Lock()


def make_pose(tf_pose):
    """
    Helper function to get a full matrix out of this pose
    """
    trans, rot = tf_pose
    pose = tra.quaternion_matrix(rot)
    pose[:3, 3] = trans
    return pose


def make_pose_from_pose_msg(msg):
    trans = (msg.pose.position.x,
             msg.pose.position.y, msg.pose.position.z,)
    rot = (msg.pose.orientation.x,
           msg.pose.orientation.y,
           msg.pose.orientation.z,
           msg.pose.orientation.w,)
    return make_pose((trans, rot))
    

class ImageListener:

    def __init__(self):

        self.cv_bridge = CvBridge()
        self.count = 0
        
        self.input_rgb = None
        self.input_depth = None
        self.input_rgb_pose = None
        self.input_stamp = None
        self.input_frame_id = None        

        # output dir
        this_dir = osp.dirname(__file__)
        self.outdir = osp.join(this_dir, '..', 'data', 'Fetch')

        # initialize a node
        rospy.init_node("image_listener")
        rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_raw', Image, queue_size=2)
        rgb_pose_sub = message_filters.Subscriber('/poserbpf_image_render_00', Image, queue_size=2)        
        depth_sub = message_filters.Subscriber('/head_camera/depth_registered/image_raw', Image, queue_size=2)        
        # depth_sub = message_filters.Subscriber('/camera/depth_registered/sw_registered/image_rect_raw', Image, queue_size=2)
        
        msg = rospy.wait_for_message('/head_camera/rgb/camera_info', CameraInfo)
        K = np.array(msg.K).reshape(3, 3)
        self.intrinsic_matrix = K
        print('Intrinsics matrix : ')
        print(self.intrinsic_matrix)        

        queue_size = 1
        slop_seconds = 0.5
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, rgb_pose_sub, depth_sub], queue_size, slop_seconds)
        ts.registerCallback(self.callback)
        
    # callback function to get images
    def callback(self, rgb, rgb_pose, depth):

        # decode image
        if depth is not None:
            if depth.encoding == '32FC1':
                depth_cv = self.cv_bridge.imgmsg_to_cv2(depth)
            elif depth.encoding == '16UC1':
                depth = self.cv_bridge.imgmsg_to_cv2(depth)
                depth_cv = depth.copy().astype(np.float32)
                depth_cv /= 1000.0
            else:
                rospy.logerr_throttle(1, 'Unsupported depth type. Expected 16UC1 or 32FC1, got {}'.format(depth.encoding))
                return
        else:
            depth_cv = None

        with lock:
            self.input_depth = depth_cv
            self.input_rgb = self.cv_bridge.imgmsg_to_cv2(rgb, 'bgr8')
            self.input_rgb_pose = self.cv_bridge.imgmsg_to_cv2(rgb_pose, 'bgr8')
            self.input_stamp = rgb.header.stamp
            self.input_frame_id = rgb.header.frame_id
        

    def save_data(self):
        # write color images
        filename = self.outdir + '/color-%06d.jpg' % self.count
        cv2.imwrite(filename, self.input_rgb )
        print(filename)
        
        # write pose color images
        filename = self.outdir + '/pose-%06d.jpg' % self.count
        cv2.imwrite(filename, self.input_rgb_pose)
        print(filename)        

        filename = self.outdir + '/depth-%06d.png' % self.count
        cv2.imwrite(filename, self.input_depth)
        print(filename)

        self.count += 1
        
        
class PoseListener:

    """
    Listens on a particular message topic.
    """

    def __init__(self, topic_name, queue_size=100, max_dt=1.0):
        self.topic_name = topic_name
        self.msg = None
        self.max_dt = max_dt
        self.last_msg_time = rospy.Time(0)
        self.reset_t = rospy.Time(0).to_sec()
        self.detections = {}
        self._sub = rospy.Subscriber(self.topic_name, DetectionList, self.callback, queue_size=queue_size)

        self.listener = tf.TransformListener()
        self.base_frame = 'base_link'
	
    def ready(self):
        if self.msg is None:
            print("[SENSOR] No message received on topic", self.topic_name)
            return False
        t = rospy.Time.now()
        dt = (t - self.last_msg_time).to_sec()
        return dt < self.max_dt

    def reset(self):
        self.detections = {}
        self.reset_t = rospy.Time.now().to_sec()
        self.msg = None

    def callback(self, msg):
        """
        Records messages coming in from perception
        """

        self.last_msg_time = rospy.Time.now()
        if (self.last_msg_time.to_sec() - self.reset_t) < self.max_dt:
            return

        self.msg = msg
        # Run through all detections in the object
        for detection in self.msg.detections:
            name = detection.name
            # print(name)
            pose = make_pose_from_pose_msg(detection.pose)
            score = detection.score
            self.detections[name] = (pose, score, self.last_msg_time)

    def get_detections(self):
        return self.detections

    def get_pose(self, obj):
        # RBPF POSE
        t = rospy.Time.now()

        if obj in self.detections:
            pose, score, last_t = self.detections[obj]

            # Check failure case from PoseCNN
            if pose[2, 3] == 0:
                return False, None

            valid = (t - last_t).to_sec() <= self.max_dt
            valid = valid and (np.sum(pose[:3, 3]) != 0)
            return valid and score > 0, pose
        else:
            return False, None


    def get_tf_pose(self, target_frame, base_frame=None, is_matrix=False):
        if base_frame is None:
            base_frame = self.base_frame
        try:
            tf_pose = self.listener.lookupTransform(base_frame, target_frame, rospy.Time(0))
            if is_matrix:
                pose = make_pose(tf_pose)
            else:
                trans, rot = tf_pose
                qt = np.zeros((4,), dtype=np.float32)
                qt[0] = rot[3]
                qt[1] = rot[0]
                qt[2] = rot[1]
                qt[3] = rot[2]
                pose = np.zeros((7, ), dtype=np.float32)
                pose[:3] = trans
                pose[3:] = qt
        except (tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException):
            pose = None
        return pose


    def get_object_poses(self, block=False):
        while 1:
            model_names = []
            object_poses = []
            object_names = []
            for name in self.detections:
                obj_name = name.split("/")[-1]
                l = len(obj_name)
                obj_name = obj_name[3:l-3]
                if obj_name == 'cabinet_handle':
                    continue

                RT = self.detections[name][0]
                pose = np.zeros((7, ), dtype=np.float32)
                pose[:3] = RT[:3, 3]
                pose[3:] = mat2quat(RT[:3, :3])
                if pose[2] == 0:
                    continue
                print(obj_name)
                print(RT)
                object_poses.append(pose)
                model_names.append(obj_name)
                object_names.append(obj_name)
            if len(object_names) > 0:
                break
            else:
                print('cannot detect object')
                if not block:
                    break
                rospy.sleep(0.5)
        print(object_names)
        return model_names, object_poses, object_names


if __name__ == '__main__':

    # image listener
    listener = ImageListener()
    
    # pose listener
    topic_name = '/poserbpf/00/info'
    pose_listener = PoseListener(topic_name)
    os.makedirs(listener.outdir, exist_ok=True)

    while 1:
        if listener.input_rgb is not None:
            print("1")
            # save object poses
            model_names, object_poses, object_names = pose_listener.get_object_poses()
            meta = {'object_names': object_names, 'poses': object_poses,
                'intrinsic_matrix': listener.intrinsic_matrix, 'factor_depth': 1000.0}
            filename = listener.outdir + '/meta-%06d.mat' % listener.count
            scipy.io.savemat(filename, meta, do_compression=True)
            print('save meta data to {}'.format(filename))            
            
            # save images
            listener.save_data()
            
            # sleep for 0.25 seconds
            break
            time.sleep(0.25)
            
            # break
