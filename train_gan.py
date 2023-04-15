import torch
import argparse
import os
from dataset import load_data, load_config
from gan import Generator, Discriminator, ReconstructionLoss

def train(cfg_file):
    cfg = load_config(cfg_file)
    train_loader, val_loader, test_loader = load_data(data_cfg=cfg["data"])
    
    # models + optimisers
    generator = Generator(4, 16, 3, 3)
    g_optimiser = torch.optim.Adam(generator.parameters(), lr=float(cfg["transformer"].get("lr")), betas=(0.9, 0.98), eps=1e-9)

    discriminator = Discriminator(4, 16, 3)
    d_optimiser = torch.optim.Adam(discriminator.parameters(), lr=float(cfg["transformer"].get("lr")), betas=(0.9, 0.98), eps=1e-9)

    # adversarial and reconstruction loss
    adv_criterion = torch.nn.BCELoss()
    rec_criterion = ReconstructionLoss()

    epochs = cfg["training"].get("epochs")

    # Save stuff
    save = cfg["training"].get("save")
    save_every = cfg["training"].get("save_every")
    save_dir = cfg["training"].get("save_dir")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        #generator.train()
        #discriminator.train()
        
        for batch_index, batch in enumerate(train_loader, 0):
            
            score, performance = batch

            discriminator.zero_grad()
            batch_size = performance.size(0)
            real_output = discriminator(performance)[:,1][-1]
            real_labels = torch.ones((batch_size, ), dtype=torch.float)
            errD_real = adv_criterion(real_output, real_labels)
            errD_real.backward(retain_graph=True)

            fake = generator(score)
            fake_labels = torch.zeros((batch_size, ), dtype=torch.float)
            fake_output = discriminator(fake)[:,1][-1]
            errD_fake = adv_criterion(fake_output, fake_labels)
            errD_fake.backward(retain_graph=True)

            d_optimiser.step()

            generator.zero_grad()
            fake_output = discriminator(fake)[:,1][-1]
            errG_adv = adv_criterion(fake_output, real_labels)
            errG_rec = rec_criterion(fake, score)
            # print(f'{errG_rec} + {errG_adv}')
            errG = errG_adv + errG_rec
            errG.backward()
            g_optimiser.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ttttt")
    parser.add_argument(
        "config",
        default="configs/asap.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    args = parser.parse_args()
    train(cfg_file=args.config)