#!/bin/bash
# FOR MPI CLUSTER USE TMPDIR=~/pip-tmp as pip cache dir
pip install -r requirements.txt   
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8%2B5043d15/pytorch3d-0.7.8%2B5043d15pt2.5.1cu124-cp310-cp310-linux_x86_64.whl
pip install numpy==1.26.4
mkdir assets/software
cd assets/software
git clone https://github.com/MPI-IS/mesh
cd mesh
make all
cd ../..
cd modules/liegroups
pip install -e .
cd ../..
pip install imageio scikit-learn scikit-image opendr psutil trimesh pyrender wandb kornia
pip install git+https://github.com/mattloper/chumpy 
