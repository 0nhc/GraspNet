import requests
import numpy as np

from PIL import Image


class GSNetFlaskClient:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self._base_url = base_url

        depth_path, intrinsics = self.DEMO_return_depth_data()
        pcd = self.DEMO_get_pcd(depth_path, intrinsics)
        grasping_pose = self.request_grasping_pose(pcd)
        print(grasping_pose)


    def request_grasping_pose(self, data):
        response = requests.post(f"{self._base_url}/get_gsnet_grasp", json=data)
        return response.json()
    

    def DEMO_return_depth_data(self):
        depth_path = "example_data/depth.png"
        intrinsics = np.array([[383.5498046875, 0.0, 320.0272521972656], 
                               [0.0, 383.5498046875, 235.78729248046875], 
                               [0.0, 0.0, 1.0]])
        return depth_path, intrinsics


    def DEMO_get_pcd(self, depth_path, intrinsics, depth_scale=1000):
        # Load depth image
        depth = np.asarray(Image.open(depth_path), dtype=np.uint16)
        depth = depth[:,:,0]

        # Load Intrinsics
        height = depth.shape[0]
        width = depth.shape[1]
        center_x = intrinsics[0, 2]
        center_y = intrinsics[1, 2]
        focal_x = intrinsics[0, 0]
        focal_y = intrinsics[1, 1]

        # Convert depth image to point cloud
        u_indices , v_indices = np.meshgrid(np.arange(width), np.arange(height))
        x_factors = (u_indices - center_x) / focal_x
        y_factors = (v_indices - center_y) / focal_y

        z_mat = depth
        x_mat = x_factors * z_mat
        y_mat = y_factors * z_mat

        points = []
        for h in range(height):
            for w in range(width):
                x = x_mat[h, w]
                y = y_mat[h, w]
                z = z_mat[h, w] / depth_scale
                points.append([x,y,z])
        
        return points
                
        
    
client = GSNetFlaskClient()