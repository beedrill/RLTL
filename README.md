# README
README for Traffic Light Optimization using DQN

## Setup environment
- virtualenv --system-site-packages DQN_ENV
    - need to inherit pysumo
- source DQN_ENV/bin/activate
- pip install --upgrade pip
- pip install --upgrade tensorflow
- if GPU
    - pip install --upgrade tensorflow-gpu
- pip install -U -r requirements.txt


## Train
```
python run_rltl.py
```
- if you run on CPU, make sure to put --cpu
- default is sumo, in order to use pysumo make sure to put --pysumo


## Evaluate
```
python run_rltl.py --mode test --load [weights.hdf5]
```