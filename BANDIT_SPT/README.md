
# HPO+SPT in the bandit case

## Dependencies
This code requires the following:

- python 3.*
- pytorch+cuda
- gym
- matplotlib
- numpy
## Usage
For example
```
python3 banditsptdemo.py  --seed 123 --classifier AM --value-method Qvalue --max-episode 10000 --learning-rate 5e-5   --log-interval 1 --envi 4x4  --margin 0.1  --flipped 0.4 --duration 5 --nepoch 300000 --sptth 0.9 --probUB 0.99 --optimizer sgd
```
