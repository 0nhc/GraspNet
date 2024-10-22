# GraspNet Reimplementation
My PC Setup:
* Ubuntu 20.04
* RTX 30/40 GPU
* CUDA 12.1

## Installation
```sh
# Create Conda Environment
conda create -n graspnet python=3.8
conda activate graspnet

# Clone this repository
git clone https://github.com/0nhc/GraspNet.git

# Install Dependencies
cd GraspNet
pip install -r requirements
# Install MinkowskiEngine
git clone https://github.com/0nhc/MinkowskiEngine.git
cd MinkowskiEngine
pip install e .
# Install PointNet2
cd ../pointnet2
python setup.py install
# Install KNN
cd ../knn
python setup.py install
# Install graspnetAPI
cd ..
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```

## Download Checkpoint
[https://drive.google.com/file/d/1drIofh4mactzU9TtWjMaLBnEJhf9HijT/view?usp=drive_link](https://drive.google.com/file/d/1drIofh4mactzU9TtWjMaLBnEJhf9HijT/view?usp=drive_link)

## Run Demo
```sh
# Terminal 1
conda activate graspnet
python flask_server.py

# Terminal 2
conda activate graspnet
python flask_client.py
```

</br>
</br>
</br>
</br>
</br>
</br>
</br>

# Original README:
# GraspNet graspness
This project aims to address issues encountered during the migration of the repository [GS-Net](https://github.com/graspnet/graspness_unofficial) to an RTX 4090 GPU.
The original repo is a fork of paper "Graspness Discovery in Clutters for Fast and Accurate Grasp Detection" (ICCV 2021) by [Zibo Chen](https://github.com/rhett-chen). 


[[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Graspness_Discovery_in_Clutters_for_Fast_and_Accurate_Grasp_Detection_ICCV_2021_paper.pdf)]
[[dataset](https://graspnet.net/)]
[[API](https://github.com/graspnet/graspnetAPI)]


## Requirements
- Python 3
- PyTorch 1.8
- Open3d 0.8
- TensorBoard 2.3
- NumPy
- SciPy
- Pillow
- tqdm
- MinkowskiEngine

## Installation
Get the code.
```bash
git clone https://github.com/graspnet/graspness_unofficial.git
cd graspness_unofficial
```
Install packages via Pip.
```bash
pip install -r requirements.txt
```
Compile and install pointnet2 operators (code adapted from [votenet](https://github.com/facebookresearch/votenet)).
```bash
cd pointnet2
python setup.py install
```
Compile and install knn operator (code adapted from [pytorch_knn_cuda](https://github.com/chrischoy/pytorch_knn_cuda)).
```bash
cd knn
python setup.py install
```
Install graspnetAPI for evaluation.
```bash
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
pip install .
```
For MinkowskiEngine, please refer https://github.com/NVIDIA/MinkowskiEngine
## Point level Graspness Generation
Point level graspness label are not included in the original dataset, and need additional generation. Make sure you have downloaded the orginal dataset from [GraspNet](https://graspnet.net/). The generation code is in [dataset/generate_graspness.py](dataset/generate_graspness.py).
```bash
cd dataset
python generate_graspness.py --dataset_root /data3/graspnet --camera_type kinect
```

## Simplify dataset
original dataset grasp_label files have redundant data,  We can significantly save the memory cost. The code is in [dataset/simplify_dataset.py](dataset/simplify_dataset.py)
```bash
cd dataset
python simplify_dataset.py --dataset_root /data3/graspnet
```

## Training and Testing
Training examples are shown in [command_train.sh](command_train.sh). `--dataset_root`, `--camera` and `--log_dir` should be specified according to your settings. You can use TensorBoard to visualize training process.

Testing examples are shown in [command_test.sh](command_test.sh), which contains inference and result evaluation. `--dataset_root`, `--camera`, `--checkpoint_path` and `--dump_dir` should be specified according to your settings. Set `--collision_thresh` to -1 for fast inference.

## Model Weights
We provide trained model weights. The model trained with RealSense data is available at [Google drive](https://drive.google.com/file/d/1RfdpEM2y0x98rV28d7B2Dg8LLFKnBkfL/view?usp=sharing) (this model is recommended for real-world application). The model trained with Kinect data is available at [Google drive](https://drive.google.com/file/d/10o5fc8LQsbI8H0pIC2RTJMNapW9eczqF/view?usp=sharing).

## Results
Results "In repo" report the model performance of my results without collision detection.

Evaluation results on Kinect camera:
|          |        | Seen             |                  |        | Similar          |                  |        | Novel            |                  | 
|:--------:|:------:|:----------------:|:----------------:|:------:|:----------------:|:----------------:|:------:|:----------------:|:----------------:|
|          | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> | __AP__ | AP<sub>0.8</sub> | AP<sub>0.4</sub> |
| In paper | 61.19  | 71.46            | 56.04            | 47.39  | 56.78            | 40.43            | 19.01  | 23.73            | 10.60             |
| In repo  | 61.83  | 73.28            | 54.14            | 51.13  | 62.53            | 41.57            | 19.94  | 24.90            | 11.02             |


## Troubleshooting
If you meet the torch.floor error in MinkowskiEngine, you can simply solve it by changing the source code of MinkowskiEngine: 
MinkowskiEngine/utils/quantization.py 262，from discrete_coordinates =_auto_floor(coordinates) to discrete_coordinates = coordinates
## Acknowledgement
My code is mainly based on Graspnet-baseline  https://github.com/graspnet/graspnet-baseline.
