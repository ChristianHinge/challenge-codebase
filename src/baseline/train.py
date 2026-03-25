import os
import torch
import yaml
import matplotlib.pyplot as plt
import torch.nn.functional as F

from monai.data import Dataset, DataLoader, CacheDataset, PersistentDataset
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

    print("Preparing dataset ...")
    dataset = PersistentDataset(
        data=data,
        transform=transforms,
        cache_dir=cfg["cache_dir"]
    )

    print("Caching dataset...")
    dataset = CacheDataset(
        data=dataset,
        cache_rate=1.0,
        num_workers=8,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
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

    print("Starting training...")

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

                loss = l1_loss(pred, y)

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