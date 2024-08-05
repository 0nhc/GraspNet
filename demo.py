import sys
import yaml
import cv2
import numpy as np

import os
import argparse
from PIL import Image
import time
import scipy.io as scio
import torch
import open3d as o3d
from graspnetAPI.graspnet_eval import GraspGroup

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import tf
import tf2_ros

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import minkowski_collate_fn
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask



class GSNet:
    def __init__(self) -> None:
        rospy.init_node("gsnet", anonymous=True)
        self._init = False
        self.poincloud_sub = rospy.Subscriber("/d435_camera/depth/points", PointCloud2, self._pointcloud_callback)
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        while not self._init:
            pass
        self._read_params()
        data, points = self._wrap_data_ros()
        grasp_group = self._inference(data)
        self._get_best_grasping_pose(grasp_group)
        self._visualize(grasp_group, points)

    def _pointcloud_callback(self, data=PointCloud2()):
        if not self._init :
            points = []
            for point in point_cloud2.read_points(data, skip_nans=True):
                x = point[0]
                y = point[1]
                z = point[2]
                points.append([x,y,z])
            self.pcd_ros = points
            self._init = True


    def _read_params(self):
        cfg_path = sys.path[0]+'/config/gsnet.yaml'
        with open(cfg_path, 'r') as config:
            self.cfg = yaml.safe_load(config)

    def _create_point_cloud_from_depth_image(self, depth, height, width, K, depth_scale):
        assert (depth.shape[0] == width and depth.shape[1] == height)
        xmap = np.arange(height)
        ymap = np.arange(width)
        xmap, ymap = np.meshgrid(xmap, ymap)
        points_z = depth.astype(np.float32) / depth_scale
        points_x = (xmap - K[0][2]) * points_z / K[0][0]
        points_y = (ymap - K[1][2]) * points_z / K[1][1]
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        return cloud
    
    def _wrap_data_ros(self):
        cloud = np.asarray(self.pcd_ros)

        mask_1 = (cloud[:,2] > 1.0) # Z in Camera Frame > 0.1m
        mask_2 = (cloud[:,2] < 1.5) # Z in Camera Frame < 1.2m
        mask_3 = (cloud[:,0] > -0.5) # X in Camera Frame > -0.7m
        mask_4 = (cloud[:,0] < 0.1) # X in Camera Frame < 0.2m
        mask = mask_1 * mask_2 * mask_3 * mask_4
        cloud_masked = cloud[mask]

        # sample points random
        if len(cloud_masked) >= self.cfg["num_point"]:
            idxs = np.random.choice(len(cloud_masked), self.cfg["num_point"], replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.cfg["num_point"] - len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]

        ret_dict = {'point_clouds': cloud_sampled.astype(np.float32),
                    'coors': cloud_sampled.astype(np.float32) / self.cfg["voxel_size"],
                    'feats': np.ones_like(cloud_sampled).astype(np.float32),
                    }
        return ret_dict, cloud_sampled
    
    def _inference(self, data_input):
        batch_data = minkowski_collate_fn([data_input])
        net = GraspNet(seed_feat_dim=self.cfg["seed_feat_dim"], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.cfg["checkpoint_path"])
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)" % (self.cfg["checkpoint_path"], start_epoch))

        net.eval()

        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data)
            grasp_preds = pred_decode(end_points)

        preds = grasp_preds[0].detach().cpu().numpy()

        gg = GraspGroup(preds)
        # collision detection
        if self.cfg["collision_thresh"] > 0:
            cloud = data_input['point_clouds']
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfg["voxel_size_cd"])
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfg["collision_thresh"])
            gg = gg[~collision_mask]

        return gg
    
    def rot2eul(self, R):
        beta = -np.arcsin(R[2,0])
        alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
        gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
        return alpha, beta, gamma
    
    def _get_best_grasping_pose(self, gg):
        gg = gg.nms()
        gg = gg.sort_by_score()
        self.best_gg = gg[0]
        for gg_to_be_slected in gg[:50]:
            _p = gg_to_be_slected.translation
            _r = gg_to_be_slected.rotation_matrix
            # Transform Matrix of the Selected Grasping Pose
            _t = np.asarray([[_r[0][0], _r[0][1], _r[0][2], _p[0]],
                             [_r[1][0], _r[1][1], _r[1][2], _p[1]],
                             [_r[2][0], _r[2][1], _r[2][2], _p[2]],
                             [0,        0,       0,        1]])
            # Extrinsic Parameters of the Camera
            _E = np.asarray(self.cfg["camera_E"])
            # Selected Grasping Pose in World Coordinate
            _tt = _E.dot(_t)
            rr, rp, ry = self.rot2eul(_tt)
            print("selecting gg, position: "+str(_tt[:3,3])+", rotation: "+str(rr)+", "+str(rp)+", "+str(ry))
            # Rules: Z > 0.05m, and 0.5 < Pitch < 0.785 rad, and Roll > 0
            if(_tt[2,3] > 0.05 and rp > 0.5 and rp < 0.785 and rr > 0):
                self.best_gg = gg_to_be_slected
                print("best gg!")
                break

        p = self.best_gg.translation
        r = self.best_gg.rotation_matrix
        t = np.asarray([[r[0][0], r[0][1], r[0][2], p[0]],
                        [r[1][0], r[1][1], r[1][2], p[1]],
                        [r[2][0], r[2][1], r[2][2], p[2]],
                        [0,0,0,1]])
        E = np.asarray(self.cfg["camera_E"])
        best_grasping_pose = E.dot(t)
        # rot_Y -90
        gripper_r_offset_1 = np.asarray([[0,0,-1,0],
                                         [0,1,0,0],
                                         [1,0,0,0],
                                         [0,0,0,1]])
        # gripper_r_offset_1 = np.asarray([[1,0,0,0],
        #                                  [0,1,0,0],
        #                                  [0,0,1,0],
        #                                  [0,0,0,1]])

        # rot_Z 180
        gripper_r_offset_2 = np.asarray([[1,0,0,0],
                                         [0,1,0,0],
                                         [0,0,1,0],
                                         [0,0,0,1]])
        # gripper_r_offset_2 = np.asarray([[-1,0,0,0],
        #                                  [0,-1,0,0],
        #                                  [0,0,1,0],
        #                                  [0,0,0,1]])
        gripper_r_offset = gripper_r_offset_1.dot(gripper_r_offset_2)

        # Pre-grasping Pose Offset
        gripper_t_offset_1 = np.asarray([[1,0,0,-self.cfg["gripper_length_offset_1"]],
                                         [0,1,0,0],
                                         [0,0,1,0],
                                         [0,0,0,1]])
        print(gripper_t_offset_1)

        # Grasping Pose Offset
        gripper_t_offset_2 = np.asarray([[1,0,0,-self.cfg["gripper_length_offset_2"]],
                                         [0,1,0,0],
                                         [0,0,1,0],
                                         [0,0,0,1]])
        print(gripper_t_offset_2)

        best_grasping_pose_1 = best_grasping_pose.dot(gripper_t_offset_1).dot(gripper_r_offset)
        best_grasping_pose_2 = best_grasping_pose.dot(gripper_t_offset_2).dot(gripper_r_offset)
        result = {}
        result["T1"] = best_grasping_pose_1.tolist()
        result["T2"] = best_grasping_pose_2.tolist()
        print(result)
        with open(self.cfg["best_grasping_pose_save_path"], 'w') as outfile:
            yaml.dump(result, outfile, default_flow_style=False)
    
    def _visualize(self, grasp_group, pointcloud):
        grippers = self.best_gg.to_open3d_geometry()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pointcloud.astype(np.float32))
        o3d.visualization.draw_geometries([cloud, grippers])
    
gsnet = GSNet()