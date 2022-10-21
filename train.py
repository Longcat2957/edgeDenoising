import os
from typing import Union
import argparse
from tqdm import tqdm
from torch.nn import MSELoss
import torch.optim as optim
from torch.utils.data import DataLoader
from libs.data_utils import *
from libs.model import UNet
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure

parser = argparse.ArgumentParser()
parser.add_argument(
    "--iteration", type=int, default=300
)
parser.add_argument(
    "--train_root", type=str, required=True
)
parser.add_argument(
    "--val_root", type=str, required=True
)
parser.add_argument(
    "--test_root", type=str
)
parser.add_argument(
    "--test_interval", type=int, default=10
)
parser.add_argument(
    "--lr", type=float, default=1e-5
)
parser.add_argument(
    "--batch_size", "-b", type=int, default=16
)
parser.add_argument(
    "--epochs", "-e", type=int, default=100
)
parser.add_argument(
    "--size", type=Union[int, tuple], default=224
)
parser.add_argument(
    "--load", type=str, default=None
)
parser.add_argument(
    "--name", type=str, required=True
)


if __name__ == "__main__":
    opt = parser.parse_args()
    print(f"# size = {opt.size}")
    savedir = os.path.join("./test", opt.name)
    if not os.path.exists("./test"):
        os.mkdir("test")
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# device = {device}")
    traindataset = trainDataset(root=opt.train_root, size=opt.size)
    valdataset = testDataset(root=opt.val_root, size=opt.size)
    trainloader = DataLoader(traindataset, batch_size=opt.batch_size, shuffle=True,
                             num_workers=os.cpu_count() -1, pin_memory=True)
    valloader = DataLoader(valdataset, batch_size=opt.batch_size, shuffle=False,
                            num_workers=os.cpu_count()-1, pin_memory=True)
    model = UNet()
    if opt.load is not None:
        try:
            model.load_state_dict(torch.load(opt.load))
        except:
            print(f"Failed to load weight >> {opt.load}")
            raise ValueError()
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, eta_min=1e-9)
    criterion = MSELoss()   # because it is continuous
    
    for epoch in range(opt.epochs):
        # train loop
        model.train()
        train_progress_bar = tqdm(iterable=trainloader, colour="green", ncols=100)
        train_loss = 0.0
        iteration = 0
        for blurredTensor, origTensor in train_progress_bar:
            iteration +=1
            blurred, gt = blurredTensor.to(device), origTensor.to(device)
            optimizer.zero_grad()
            denoised = model(blurred)
            loss = criterion(denoised, gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_progress_bar.set_description(
                f"# [TRAIN]({epoch}/{opt.epochs}) | loss = {(train_loss / iteration):.5f}"
            )
            if iteration > opt.iteration:
                break

        model.eval()
        val_progress_bar = tqdm(iterable=valloader, colour="blue", ncols=100)
        with torch.no_grad():
            iteration = 0
            sum_psnr, sum_ssnr = 0.0, 0.0
            for test_tensor in val_progress_bar:
                iteration += 1
                test_tensor = test_tensor.to(device)
                denoised = model(test_tensor)
                psnr = peak_signal_noise_ratio(denoised, test_tensor)
                ssnr = structural_similarity_index_measure(denoised, test_tensor)
                
                sum_psnr += psnr.item()
                sum_ssnr += ssnr.item()
                
                val_progress_bar.set_description(
                    f"# [VAL]({epoch}/{opt.epochs}) | psnr = {(sum_psnr / iteration):.3f}, ssnr = {(sum_ssnr / iteration):.3f}"
                )
        scheduler.step()

    # after train