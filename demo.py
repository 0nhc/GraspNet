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
        self.poincloud_sub = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self._pointcloud_callback)

        while not self._init:
            pass
        self._read_params()
        data, points = self._wrap_data_ros()
        grasp_group = self._inference(data)
        self._visualize(grasp_group, points)

        rospy.spin()

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

        mask_1 = (cloud[:,2] > 0)
        mask_2 = (cloud[:,2] < 1.5)
        mask_3 = (cloud[:,0] > -0.05)
        mask_4 = (cloud[:,0] < 0.6)
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
    
    def _wrap_data(self):
        # Load Data
        depth = cv2.imread(self.cfg["depth_img_path"])
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        width = self.cfg["camera_width"]
        height = self.cfg["camera_height"]
        K = self.cfg["camera_K"]
        depth_scale = self.cfg["depth_scale"]

        # Generate Pointcloud
        cloud = self._create_point_cloud_from_depth_image(depth, height, width, K, depth_scale)

        # Sample Points
        mask_1 = (cloud[:,:,2] > 0)
        mask_2 = (cloud[:,:,2] < 1.5)
        mask_3 = (cloud[:,:,0] > -0.05)
        mask_4 = (cloud[:,:,0] < 0.6)
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
    
    def _visualize(self, grasp_group, pointcloud):
        grasp_group = grasp_group.nms()
        grasp_group = grasp_group.sort_by_score()
        if grasp_group.__len__() > 30:
            grasp_group = grasp_group[:30]
        grippers = grasp_group.to_open3d_geometry_list()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pointcloud.astype(np.float32))
        o3d.visualization.draw_geometries([cloud, *grippers])
    
gsnet = GSNet()