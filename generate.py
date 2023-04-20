import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import shutil
from dataset import load_data, load_config
from gan import Generator, Discriminator, ReconstructionLoss
from tqdm import tqdm
from extract import convert

GEN_PT = 'gen_epoch_80.pt'
DIS_PT = 'dis_epoch_80.pt'

# In most cases you shouldn't use this. There's no guarantee the model hasn't seen the data pulled from the testloader, because
# it isn't the same testloader generated when it was trained.
def _generate(cfg_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    cfg = load_config(cfg_file)
    _, _, test_loader = load_data(data_cfg=cfg["data"])

    save_dir = cfg["training"].get("save_dir")
    
    # models
    generator = Generator(4, 16, 3, 3).to(device)
    generator.load_state_dict(torch.load(os.path.join(save_dir, GEN_PT), map_location=device))

    score, _, score_path, _ = next(iter(test_loader))
    print(score)
    print(f'Generating new score from {score_path[0]}...')
    gen_path = '/'.join(score_path[0].split('/')[:-1])
    latent = torch.randn(1, 16, 4)
    gen_input = torch.cat((latent, score), dim=1)
    fake = generator(gen_input)[:,16:]
    fake = fake.detach().numpy()[0]
    convert(fake, gen_path)
    print(f'Done!')

def generate(generator, score, score_path):
    print(f'Generating new score from {score_path[0]}...')
    gen_path = '/'.join(score_path[0].split('/')[:-1])
    latent = torch.randn(1, 16, 4)
    gen_input = torch.cat((latent, score), dim=1)
    fake = generator(gen_input)[:,16:]
    fake = fake.detach().numpy()[0]
    convert(fake, gen_path)
    print(f'Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("ttttt")
    parser.add_argument(
        "config",
        default="configs/asap.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    args = parser.parse_args()
    _generate(cfg_file=args.config)