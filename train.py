import torch
import os
from dataset import load_data, load_config
from model import Generator, Discriminator

def train(cfg_file):
    cfg = load_config(cfg_file)
    train_loader, val_loader, test_loader = load_data(data_cfg=cfg["data"])
    
    # models + optimisers + loss
    generator = Generator()
    g_optimiser = torch.optim.Adam(generator.parameters(), lr=float(cfg["transformer"].get("lr")), betas=(0.9, 0.98), eps=1e-9)
    g_criterion = torch.nn.MSELoss()

    discriminator = Discriminator()
    d_optimiser = torch.optim.Adam(discriminator.parameters(), lr=float(cfg["transformer"].get("lr")), betas=(0.9, 0.98), eps=1e-9)
    d_criterion = torch.nn.BCELoss()

    epochs = cfg["training"].get("epochs")

    # Save stuff
    save = cfg["training"].get("save")
    save_every = cfg["training"].get("save_every")
    save_dir = cfg["training"].get("save_dir")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        generator.train()
        discriminator.train()
        
        for batch_index, batch in enumerate(train_loader):
            
            score, performance = batch
            discriminator.zero_grad()
            batch_size = performance.size(0)
            real_output = discriminator(performance)
            real_labels = torch.ones((batch_size, ), 1, dtype=torch.float)
            errD_real = d_criterion(real_output, real_labels)
            errD_real.backward()

            fake = generator(score)
            fake_labels = torch.zeros((batch_size, ), 1, dtype=torch.float)
            fake_output = discriminator(fake)
            errD_fake = d_criterion(fake_output, fake_labels)
            errD_fake.backward()

            d_optimiser.step()

            generator.zero_grad()
            fake_output = discriminator(fake)
            errG = g_criterion(fake_output, real_labels)
            errG.backward()
            g_optimiser.step()

