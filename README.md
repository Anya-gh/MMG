## MMG

A music generation model that uses a note-matching approach to transform a score into a performacne with human dynamics and tempo. The matching algorithm is in ```extract.py```, which computes a least-cost mapping between two sequences. A simple LSTM based GAN is implemented to test this approach, which is trained on the [ASAP dataset](https://github.com/fosfrancesco/asap-dataset). The dataset loader is in ```dataset.py```.

### Setup

#### Usage

Running ```run.sh``` will create a virtual environment, install the required libraries, start training the GAN and start tensorboard. By default it will use the first available GPU, or the CPU if there is no GPU.

If you want to run just the training script, use ```python train_gan.py asap.yaml```. The config file, ```asap.yaml``` can be changed, or replaced entirely, with any hyperparameters of your choice; the one given is for ASAP, so configure any replacement files accordingly for the dataset used.

#### Data

Use ```download.sh``` to get the [ASAP dataset](https://github.com/fosfrancesco/asap-dataset).

