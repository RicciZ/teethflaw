## Environment
```bash
# python env
conda create -n teethflaw python=3.8
conda activate teethflaw
# mesh boolean lib
sudo apt-get install openscad
# mesh lib
conda install -c conda-forge trimesh
conda install rtree
# 3rd-party libs
conda install -c anaconda tqdm
# torch
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch # use cuda11 in our server, so change this accordingly
# tensorboardX
conda install -c conda-forge tensorboardx
conda install tensorflow-gpu cudatoolkit=10.2 (10.2 on PC, 11.1 on server)
# win -> unix (on PC if needed)
conda install -c msys2 m2-base
# install if needed
conda install -c conda-forge h5py
conda install -c conda-forge plyfile
```

## Hole Prediction

### Preprocess
```bash
cd teethflaw/preprocess
# to test locally we use sample with tooth id = 11
# on the server we can try --all True
# sample data are on the ../data/bad
# all data are on the /rsch/teethhole on the server
# preprocess.py line 25 as_dcm() (scipy 1.2.1) -> as_matrix() (scipy 1.6.2 new version)

python preprocess.py --data ../data/sample --save ../data/std_data --id 11
python split.py --data ../data/std_data --save ../data/std_split_data --id 11
python dataset_stat.py --id 11
```

### Train
```bash
cd teethflaw/hole_prediction
# to test locally we use sample with tooth id = 11
# on the server we can try --all True
# epoch default 200, batch size default 4, to test locally, decrease to 1 and 2
python main.py --loss weighted_ce_var_i --stat ../data/stat_11.json --opt adam
python main.py --loss weighted_ce --stat ../data/stat_11.json
python main.py --loss focal_loss --alpha 0.1 --gamma 2
```

### Test
```bash
python test.py --model_path []
python visulization.py --path []
```

