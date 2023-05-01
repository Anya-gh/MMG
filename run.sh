#!/bin/bash
python -m venv env
source env/bin/activate
python -m pip install --upgrade pip
pip3 install torch tensorboard pandas pretty_midi pyyaml tqdm protobuf==3.19.6
python -m tensorboard.main --logdir=log & python train_gan.py asap.yaml