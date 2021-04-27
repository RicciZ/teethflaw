## Environment
```bash
# python env
conda create -n teethflaw python=3.8
conda activate teethflaw
# mesh boolean lib
sudo apt-get install openscad
# mesh lib
conda install -c conda-forge trimesh
conda install -c conda-forge rtree
# 3rd-party libs
conda install -c anaconda tqdm
# torch
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
# use cuda11 in our server, so change this accordingly
# tensorboardX
conda install -c conda-forge tensorboardx
conda install tensorflow-gpu cudatoolkit=11.1
# win -> unix (on PC if needed)
conda install -c msys2 m2-base
# install if needed
conda install -c conda-forge h5py
conda install -c conda-forge plyfile
conda install -c anaconda scipy
conda install -c conda-forge pandas
conda install -c anaconda networkx
```

## Server
### open notebook on the server via browser
```bash
# 1. open notebook on the server
#    --ip=0.0.0.0 means visit with any ip
#    --no-browser means open without browser
#    --port=XXXX open with specific port you want
jupyter notebook --ip=0.0.0.0 --no-browser --port=XXXX

# 2. connect local port YYYY to the server port XXXX
ssh -N -f -L localhost:YYYY:localhost:XXXX remoteuser@remotehost

# 3. visit localhost:YYYY via broser locally
```

## TMUX
|      shortcut key & command            |      function                                |
|:--------------------------------------:|:--------------------------------------------:|
| Ctrl + b + [                           | to see history (q to exit)                   |
| Ctrl + b + d                           | detach                                       |
| Ctrl + d                               | directly exit and kill current session       |
| Ctrl + b + s                           | list the sessions                            |
| Ctrl + b + $                           | rename current session                       |
| Ctrl + b + %                           | split window left and right                  |
| Ctrl + b + "                           | split window up and down                     |
| Ctrl + b + <arrow key>                 | switch cursor to other panes                 |
| Ctrl + b + x                           | close current pane                           |
| Ctrl + b + q                           | display pane number                          |
| Ctrl + b + $                           | rename current session                       |
| Ctrl + b + %                           | split window left and right                  |
| tmux ls                                | see the tmux info                            |
| tmux new -s <session-name>             | create session                               |
| tmux a -t <session-name>               | attach to a session                          |
| tmux kill-session -t <session-name>    | kill a session                               |




## Hole Prediction
### GPU Use
```bash
# check GPU info
nvidia-smi
# Specify the GPU we want to use 
# e.g. to use GPU no. 2 to run train.py
CUDA_VISIBLE_DEVICES=2 python train.py
```

### Preprocess
```bash
cd teethflaw/preprocess
# to test locally we use sample with tooth id = 11
# on the server we can try --all True
# sample data are on the ../data/bad
# all data are on the /rsch/teethhole on the server
# preprocess.py line 25 as_dcm() (scipy 1.2.1) -> as_matrix() (scipy 1.6.2 new version)

python preprocess.py --data ../data/bad --save ../data/std_data --id 11
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

