# README
README for Traffic Light Optimization using DQN

This branch is to adapt SUMO 0.26.0 for testing LuST environment

## Setup environment
- virtualenv --system-site-packages DQN_ENV
    - need to inherit pysumo
- source DQN_ENV/bin/activate
- pip install --upgrade pip
- pip install --upgrade tensorflow
- if GPU
    - pip install --upgrade tensorflow-gpu
- pip install -U -r requirements.txt


## Single Agent
### Train
```
python run_rltl.py
```
- if you run on CPU, make sure to put --cpu
- default is sumo, in order to use pysumo make sure to put --pysumo

### Evaluate
```
python run_rltl.py --mode test --load [weights.hdf5]
```

## Multi Agents
### Train
```
python run_multiagents_rltl.py
```
- if you run on CPU, make sure to put --cpu
- default is sumo, in order to use pysumo make sure to put --pysumo

### Evaluate
```
python run_multiagents_rltl.py --mode test --load [weights.hdf5]
```
- currently, weights are saved as ~weights_[agent id].hdf5, so when you load, remove letters after weights
    - e.g. DQN_SUMO_best_weights_0.hdf5 => DQN_SUMO_best_weights
