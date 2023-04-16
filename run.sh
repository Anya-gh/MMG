python -m venv env
source env/bin/activate
pip3 install torch tensorboard pandas pretty_midi pyyaml tqdm protobuf==3.19.6
tensorboard --logdir=log & python train_gan.py asap.yaml