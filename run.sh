python -m venv env
source env/bin/activate
pip install torch
pip install tensorboard
pip install pandas
pip install pretty_midi
pip install pyyaml
pip install tqdm
pip install protobuf==3.19.6
python train_gan.py asap.yaml