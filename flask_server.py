import sys
import yaml
import os


from flask import Flask, request, jsonify
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import minkowski_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask


class GSNetFlaskServer:
    def __init__(self):
        # Read Params
        self._read_params()

        # Flask related
        self.app = Flask(__name__)
        self.setup_routes()

        # GSNet related

    
    
    def setup_routes(self):
        @self.app.route('/get_gsnet_grasp', methods=['POST'])
        def get_gsnet_grasp():
            data = request.json
            return jsonify(data)
    

    def run(self):
        self.app.run(debug=True)


    # def _gsnet_inference(self, data_input):
    #     batch_data = minkowski_collate_fn([data_input])
    #     net = GraspNet(seed_feat_dim=self.cfg["seed_feat_dim"], is_training=False)
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     net.to(device)
    #     # Load checkpoint
    #     checkpoint = torch.load(self.cfg["checkpoint_path"])
    #     net.load_state_dict(checkpoint['model_state_dict'])
    #     start_epoch = checkpoint['epoch']
    #     print("-> loaded checkpoint %s (epoch: %d)" % (self.cfg["checkpoint_path"], start_epoch))

    #     net.eval()

    #     for key in batch_data:
    #         if 'list' in key:
    #             for i in range(len(batch_data[key])):
    #                 for j in range(len(batch_data[key][i])):
    #                     batch_data[key][i][j] = batch_data[key][i][j].to(device)
    #         else:
    #             batch_data[key] = batch_data[key].to(device)
    #     # Forward pass
    #     with torch.no_grad():
    #         end_points = net(batch_data)
    #         grasp_preds = pred_decode(end_points)

    #     preds = grasp_preds[0].detach().cpu().numpy()

    #     gg = GraspGroup(preds)
    #     # collision detection
    #     if self.cfg["collision_thresh"] > 0:
    #         cloud = data_input['point_clouds']
    #         mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfg["voxel_size_cd"])
    #         collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfg["collision_thresh"])
    #         gg = gg[~collision_mask]

    #     return gg
    

    def _read_params(self):
        cfg_path = sys.path[0]+'/config/gsnet.yaml'
        with open(cfg_path, 'r') as config:
            self.cfg = yaml.safe_load(config)


gsnetflaskserver = GSNetFlaskServer()
gsnetflaskserver.run()
