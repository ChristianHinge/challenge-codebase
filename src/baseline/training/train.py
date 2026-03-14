import os
import torch
import yaml

from monai.data import CacheDataset, DataLoader
from tqdm import tqdm

from datasets.dataset import get_dataset
from datasets.transforms import get_train_transforms
from models.unet import build_model


# enable cudnn autotuner
torch.backends.cudnn.benchmark = True


def load_config():

    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def main():

    cfg = load_config()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Using device:", device)

    data = get_dataset(cfg["data_dir"])

    transforms = get_train_transforms(
        cfg["patch_size"],
        cfg["spacing"],
    )

    # cache dataset for faster loading
    dataset = CacheDataset(
        data=data,
        transform=transforms,
        cache_rate=1.0,
        num_workers=cfg["num_workers"],
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        persistent_workers=True,
    )

    model = build_model().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
    )

    loss_fn = torch.nn.L1Loss()

    scaler = torch.cuda.amp.GradScaler()

    os.makedirs("outputs/checkpoints", exist_ok=True)

    for epoch in range(cfg["epochs"]):

        model.train()

        epoch_loss = 0

        pbar = tqdm(loader)

        for batch in pbar:

            x = batch["input"].to(device)
            y = batch["ct"].to(device)

            optimizer.zero_grad()

            # mixed precision
            with torch.cuda.amp.autocast():

                pred = model(x)
                loss = loss_fn(pred, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            pbar.set_description(f"loss {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)

        print("Epoch", epoch, "Loss", avg_loss)

        torch.save(
            model.state_dict(),
            f"outputs/checkpoints/model_epoch_{epoch}.pth",
        )


if __name__ == "__main__":
    main()