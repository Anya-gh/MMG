import torch
import argparse
import os
from dataset import load_data, load_config
from lstm import LSTM

def train(cfg_file):
    cfg = load_config(cfg_file)
    train_loader, val_loader, test_loader = load_data(data_cfg=cfg["data"])
    
    # models + optimisers + loss
    lstm = LSTM(4, 16, 3, 3)
    optimiser = torch.optim.Adam(lstm.parameters(), lr=float(cfg["transformer"].get("lr")), betas=(0.9, 0.98), eps=1e-9)
    criterion = torch.nn.MSELoss()

    epochs = cfg["training"].get("epochs")

    # Save stuff
    save = cfg["training"].get("save")
    save_every = cfg["training"].get("save_every")
    save_dir = cfg["training"].get("save_dir")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for epoch in range(epochs):
        print(f'Epoch: {epoch}')
        for batch_index, batch in enumerate(train_loader, 0):          
            score, performance, _, _ = batch
            lstm.zero_grad()
            output = lstm(score)
            loss = criterion(output, performance)
            loss.backward()
            optimiser.step()

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