import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 5
NUM_WORKERS = 1
IMAGE_HEIGHT = 512  # 512
IMAGE_WIDTH = 512  # 512
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_GLYCOGEN = False
LOAD_ASTROCYTE = False
TRAIN_IMG_DIR = "/Users/yananw/Desktop/588FinalProject/data/astro3/train"
TRAIN_MASK_DIR = "/Users/yananw/Desktop/588FinalProject/data/astro3/mask"
VAL_IMG_DIR = "/Users/yananw/Desktop/588FinalProject/data/val/train"
VAL_MASK_DIR = "/Users/yananw/Desktop/588FinalProject/data/val/mask"
CLASS_WEIGHT = torch.Tensor([0.3,0.7]) #default 1,1,1 affects crossentropyloss

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.unsqueeze(1).to(device=DEVICE)
        #targets = targets.long().to(device=DEVICE)
        #change targets type to int 64

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            #changes all predictions to 0 0 1 2 2 types
            #      predictions = torch.argmax(predictions, dim=1).float()
            # predictions = torch.argmax(predictions,keepdim=True).float()
            #but what are the saved images look like? out channels = 3. what does this represent?
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],  #can I change to size 1 not size 3 as my inputs are grayscale
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    #I changed out_channels to be 3. What are the impacts?
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    # can add class weight to entropyLoss
    #loss_fn = nn.CrossEntropyLoss(weight = CLASS_WEIGHT)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    #check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    if LOAD_MODEL:
        if LOAD_GLYCOGEN:
            load_checkpoint(torch.load("glycogen.pth.tar"), model)
        elif LOAD_ASTROCYTE:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

        check_accuracy(val_loader, model, device=DEVICE)
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )



    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()