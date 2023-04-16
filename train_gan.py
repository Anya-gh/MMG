import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import shutil
from dataset import load_data, load_config
from gan import Generator, Discriminator, ReconstructionLoss
from tqdm import tqdm

def train(cfg_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    cfg = load_config(cfg_file)
    train_loader, val_loader, test_loader = load_data(data_cfg=cfg["data"])
    
    # models + optimisers
    generator = Generator(4, 16, 3, 3).to(device)
    g_optimiser = torch.optim.Adam(generator.parameters(), lr=float(cfg["transformer"].get("lr")), betas=(0.9, 0.98), eps=1e-9)

    discriminator = Discriminator(4, 16, 3).to(device)
    d_optimiser = torch.optim.Adam(discriminator.parameters(), lr=float(cfg["transformer"].get("lr")), betas=(0.9, 0.98), eps=1e-9)

    # adversarial and reconstruction loss
    adv_criterion = torch.nn.BCELoss()
    rec_criterion = ReconstructionLoss()

    epochs = cfg["training"].get("epochs")

    log_dir = cfg["generic"].get("log_dir")
    if cfg['generic'].get('clear_log') and os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

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
        
        for batch_idx, batch in tqdm(enumerate(train_loader)):
            
            score, performance = batch
            score, performance = score.to(device), performance.to(device)
            if torch.cuda.is_available():
                score, performance = score.cuda(), performance.cuda()

            discriminator.zero_grad()
            batch_size = performance.size(0)
            real_output = discriminator(performance)[:,1][-1]
            real_labels = torch.ones((batch_size, ), dtype=torch.float)
            errD_real = adv_criterion(real_output, real_labels)
            errD_real.backward(retain_graph=True)

            latent = torch.randn(batch_size, 16, 4)
            gen_input = torch.cat((latent, score), dim=1)
            # Don't care about the output for the latent vector
            fake = generator(gen_input)[:,16:]
            fake_labels = torch.zeros((batch_size, ), dtype=torch.float)
            fake_output = discriminator(fake)[:,1][-1]
            errD_fake = adv_criterion(fake_output, fake_labels)
            errD_fake.backward(retain_graph=True)

            errD = errD_real + errD_fake

            d_optimiser.step()

            generator.zero_grad()
            fake_output = discriminator(fake)[:,1][-1]
            errG_adv = adv_criterion(fake_output, real_labels)
            errG_rec = rec_criterion(fake, score)
            # print(f'{errG_rec} + {errG_adv}')
            errG = errG_adv + errG_rec
            errG.backward()
            g_optimiser.step()

            global_step = epoch*len(train_loader) + batch_idx
            writer.add_scalar('(Generator) Loss/train', errG.item(), global_step)
            writer.add_scalar('(Discriminator) Loss/train', errD.item(), global_step)

        generator.eval()
        errG_val = 0
        errD_val = 0
        for batch_idx, batch in tqdm(enumerate(val_loader)):
            with torch.no_grad():
                score, performance = batch

                batch_size = performance.size(0)
                real_output = discriminator(performance)[:,1][-1]
                real_labels = torch.ones((batch_size, ), dtype=torch.float)
                errD_real = adv_criterion(real_output, real_labels)

                fake = generator(score)
                fake_labels = torch.zeros((batch_size, ), dtype=torch.float)
                fake_output = discriminator(fake)[:,1][-1]
                errD_fake = adv_criterion(fake_output, fake_labels)

                errD_val += errD_real + errD_fake

                fake_output = discriminator(fake)[:,1][-1]
                errG_adv = adv_criterion(fake_output, real_labels)
                errG_rec = rec_criterion(fake, score)
                errG_val += errG_adv + errG_rec

        # Tensorboard
        errG_avg = errG_val / len(val_loader)
        errD_avg = errD_val / len(val_loader)
        writer.add_scalar('(Generator) Loss/val', errG_avg, global_step)
        writer.add_scalar('(Discriminator) Loss/val', errD_avg, global_step)

        if save and epoch % save_every == 0:
            gen_save_loc = os.path.join(save_dir, "gen_epoch_{:}.pt".format(epoch))
            torch.save(generator.state_dict(), gen_save_loc)
            dis_save_loc = os.path.join(save_dir, "dis_epoch_{:}.pt".format(epoch))
            torch.save(discriminator.state_dict(), dis_save_loc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("ttttt")
    parser.add_argument(
        "config",
        default="configs/asap.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="gpu to run your job on"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train(cfg_file=args.config)