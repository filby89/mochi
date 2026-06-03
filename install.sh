#!/bin/bash
# Pin numpy<2 for every pip install in this script. The MPI-IS `mesh` build below
# (`make all` runs `pip install --upgrade -r requirements.txt`) otherwise pulls
# numpy 2.x, which fails to compile mesh's C++ against the numpy-2 headers.
CONSTRAINTS="$(mktemp)"
echo "numpy==1.26.4" > "$CONSTRAINTS"
export PIP_CONSTRAINT="$CONSTRAINTS"

pip install -r requirements.txt
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install https://github.com/MiroPsota/torch_packages_builder/releases/download/pytorch3d-0.7.8%2B5043d15/pytorch3d-0.7.8%2B5043d15pt2.5.1cu121-cp310-cp310-linux_x86_64.whl
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121/index.html
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

# -----------------------------------------------------------------------------
# FLAME 2023 model (license-restricted — requires a free FLAME account)
# -----------------------------------------------------------------------------
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

if [ ! -f assets/FLAME2023/flame2023_no_jaw.pkl ]; then
    echo -e "\nDownloading the FLAME 2023 model."
    echo "Register at https://flame.is.tue.mpg.de/ and agree to the license first."
    read -p "Username (FLAME): " username
    read -p "Password (FLAME): " password
    username=$(urle "$username")
    password=$(urle "$password")
    mkdir -p assets/FLAME2023
    wget --post-data "username=$username&password=$password" \
        'https://download.is.tue.mpg.de/download.php?domain=flame&resume=1&sfile=FLAME2023.zip' \
        -O assets/FLAME2023/FLAME2023.zip --no-check-certificate --continue
    unzip -o assets/FLAME2023/FLAME2023.zip -d assets/FLAME2023/
    # The archive nests everything under a top-level FLAME2023/ folder; flatten it so the
    # model lands at assets/FLAME2023/flame2023_no_jaw.pkl (the path the code expects).
    if [ -d assets/FLAME2023/FLAME2023 ]; then
        mv assets/FLAME2023/FLAME2023/* assets/FLAME2023/
        rmdir assets/FLAME2023/FLAME2023
    fi
    rm -f assets/FLAME2023/FLAME2023.zip
    echo "FLAME 2023 model downloaded to assets/FLAME2023/."
else
    echo "FLAME 2023 model already present in assets/FLAME2023/, skipping download."
fi
