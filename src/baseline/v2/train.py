import os
import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F

from monai.data import PersistentDataset, DataLoader
from tqdm import tqdm

from dataset import get_dataset
from transforms import get_train_transforms
from unet import build_model


torch.backends.cudnn.benchmark = True


# -----------------------------
# CONFIG
# -----------------------------

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


# -----------------------------
# CT GRADIENT LOSS
# -----------------------------

def gradient_loss(pred, target):

    dx_pred = torch.abs(pred[:,:,1:,:,:] - pred[:,:,:-1,:,:])
    dx_gt   = torch.abs(target[:,:,1:,:,:] - target[:,:,:-1,:,:])

    dy_pred = torch.abs(pred[:,:,:,1:,:] - pred[:,:,:,:-1,:])
    dy_gt   = torch.abs(target[:,:,:,1:,:] - target[:,:,:,:-1,:])

    dz_pred = torch.abs(pred[:,:,:,:,1:] - pred[:,:,:,:,:-1])
    dz_gt   = torch.abs(target[:,:,:,:,1:] - target[:,:,:,:,:-1])

    return (
        F.l1_loss(dx_pred, dx_gt)
        + F.l1_loss(dy_pred, dy_gt)
        + F.l1_loss(dz_pred, dz_gt)
    )


# -----------------------------
# TRAIN
# -----------------------------

def main():

    cfg = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    data = get_dataset(cfg["data_dir"])

    transforms = get_train_transforms(
        cfg["patch_size"],
    )

    print("Preparing dataset cache...")

    dataset = PersistentDataset(
        data=data,
        transform=transforms,
        cache_dir=os.path.dirname(cfg["data_dir"]) + "/.cache"
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    model = build_model().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"]
    )

    l1_loss = torch.nn.L1Loss()

    scaler = torch.amp.GradScaler("cuda")

    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs("outputs/plots", exist_ok=True)

    best_loss = float("inf")

    loss_history = []

    for epoch in range(cfg["epochs"]):

        model.train()

        epoch_loss = 0

        pbar = tqdm(loader)

        for batch in pbar:

            x = batch["input"].to(device)
            y = batch["ct"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):

                pred = model(x)

                l1 = l1_loss(pred, y)
                grad = gradient_loss(pred, y)

                loss = l1 + 0.2 * grad

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            pbar.set_description(f"loss {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)

        print("Epoch", epoch, "Loss", avg_loss)

        loss_history.append(avg_loss)

        scheduler.step()

        # best checkpoint
        if avg_loss < best_loss:

            best_loss = avg_loss

            torch.save(
                model.state_dict(),
                "outputs/checkpoints/best_model.pth"
            )

        # last checkpoint
        torch.save(
            model.state_dict(),
            "outputs/checkpoints/last_model.pth"
        )

        # log
        with open("outputs/logs/train_log.txt","a") as f:
            f.write(f"{epoch},{avg_loss}\n")

        # plot loss
        plt.figure()
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig("outputs/plots/loss_curve.png")
        plt.close()


if __name__ == "__main__":
    main()