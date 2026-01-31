"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
The script trains a model on a specified dataset of the MedMNIST+ collection and saves the best performing model.
"""

# Import packages
import optuna
import os 
import argparse
import yaml
import torch
import torch.nn as nn
import timm
import time
import medmnist
import random
import numpy as np
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast

from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
import torch.optim as optim

from timm.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import alexnet, AlexNet_Weights
from medmnist import INFO
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

# Import custom modules
from utils import calculate_passed_time, seed_worker, get_ACC, get_AUC

#from transformers import AutoModel
from safetensors.torch import load_file


def train(config: dict, train_loader: DataLoader, val_loader: DataLoader):
    """
    Train a model on the specified dataset and save the best performing model.

    :param config: Dictionary containing the parameters and hyperparameters.
    :param train_loader: DataLoader for the training set.
    :param val_loader: DataLoader for the validation set.
    """

    # Start code
    start_time = time.time()
    print("\tStart training ...")

    # Create the model
    print("\tCreate the model ...")
    if config['architecture'] == 'alexnet':
        model = alexnet(weights=AlexNet_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(4096, config['num_classes'])
    else:
        model = timm.create_model(config['architecture'], pretrained=True, num_classes=config['num_classes'], drop_rate=0.4, drop_path_rate=0.2)

    # Initialize the model for the given training procedure
    print("\tInitialize the model for the given training procedure ...")
    if config['training_procedure'] == 'endToEnd':
        pass

    elif config['training_procedure'] == 'linearProbing':
        # Set only the last layer to trainable
        for param in model.parameters():
            param.requires_grad = False

        if config['architecture'] == 'alexnet':
            for param in model.classifier[6].parameters():
                param.requires_grad = True
        else:
            for param in model.get_classifier().parameters():
                param.requires_grad = True

    else:
        raise ValueError("Training procedure not supported.")


     # Move the model to the available device
    model = model.to(config['device'])

    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name or "head" in name or "fc" in name:
            decoder_params.append(param)
        else:
            encoder_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": encoder_params, "lr": config['lr_encoder']},
            {"params": decoder_params, "lr": config['lr_decoder']},
        ],
        weight_decay=config['weight_decay']
    )

    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

    warmup_epochs = 10
    total_epochs = config["epochs"]

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,   # start at 10% LR
        total_iters=warmup_epochs
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=total_epochs - warmup_epochs,
        eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )

    scaler = torch.cuda.amp.GradScaler()

    # Define the loss function
    print("\tDefine the loss function ...")
    if config['task'] == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss().to(config['device'])
        prediction = nn.Sigmoid()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing']) #CEL with Label Smoothing 
        prediction = nn.Softmax(dim=1)

    # Create variables to store the best performing model
    print("\tInitialize helper variables ...")
    best_val_loss, best_epoch = np.inf, 0
    val_acc = 0
    best_val_acc = 0
    best_model = deepcopy(model)
    epochs_no_improve = 0  # Counter for epochs without improvement
    n_epochs_stop = 100  # Number of epochs to wait before stopping

    history = {
        "train_acc": [],
        "train_loss": [],
        "val_acc": [],
        "val_loss": [],
        "val_auc": []
    }

    # Training loop
    print(f"\tRun the training for {config['epochs']} epochs ...")
    for epoch in range(config['epochs']):
        start_time_epoch = time.time()  # Stop the time
        print(f"\t\tEpoch {epoch} of {config['epochs']}:")

        # Training
        print(f"\t\t\t Train:")
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        # Training Loop for this trial
        for images, labels in tqdm(train_loader):
            # Map the data to the available device
            images = images.to(config['device'])

            if config['task'] == 'multi-label, binary-class':
                labels = labels.to(torch.float32).to(config['device'])
            else:
                labels = torch.squeeze(labels, 1).long().to(config['device'])


            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)   
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

            # ---- TRAIN ACCURACY ----
            if config['task'] == 'multi-label, binary-class':
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                correct_per_sample = preds.eq(labels).all(dim=1)  # (B,)
                train_correct += correct_per_sample.sum().item()
                train_total += labels.size(0)
            else:
                preds = outputs.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

        # Update the learning rate
        scheduler.step()

        # -----------------
        # VALIDATION
        # -----------------
        print(f"\t\t\t Evaluate:")
        model.eval()
        val_loss = 0.0

        y_true_list = []
        y_logits_list = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader):
                images = images.to(config['device'])
                outputs = model(images)

                # Run the forward pass
                if config['task'] == 'multi-label, binary-class':
                    labels = labels.to(torch.float32).to(config['device'])
                    loss = criterion(outputs, labels)
                    outputs = prediction(outputs).to(config['device'])

                else:
                    labels = torch.squeeze(labels, 1).long().to(config['device'])
                    loss = criterion(outputs, labels)
                    outputs = prediction(outputs).to(config['device'])
                    labels = labels.float().resize_(len(labels), 1)


                # Store the current loss
                val_loss += loss.item()

                # store for metrics (detach + move to CPU)
                y_true_list.append(labels.cpu())
                y_logits_list.append(outputs.cpu())

        # concatenate
        y_true = torch.cat(y_true_list, dim=0)
        y_logits = torch.cat(y_logits_list, dim=0)

        # Calculate the metrics
        if config['task'] == 'multi-label, binary-class':
            y_prob = torch.sigmoid(y_logits)
            y_pred = (y_prob > 0.5).float()
        else:
            y_prob = torch.softmax(y_logits, dim=1)
            y_pred = y_logits.argmax(dim=1)

        val_acc = get_ACC(y_true.numpy(), y_pred.numpy(), config['task'])
        val_auc = get_AUC(y_true.numpy(), y_prob.numpy(), config['task'])


        # Print the loss values and send them to wandb
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total

        print(f"\t\t\tTrain ACC: {train_acc}")
        print(f"\t\t\tTrain Loss: {train_loss}")
        print(f"\t\t\tVal ACC: {val_acc}")
        print(f"\t\t\tVal Loss: {val_loss}")
        print(f"\t\t\tVal AUC: {val_auc}")

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        # Store the current best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model = deepcopy(model)
            epochs_no_improve = 0  # Reset the counter
        else:
            epochs_no_improve += 1  # Increment the counter
        print(f"\t\t\tCurrent best Val Loss: {best_val_loss}")
        print(f"\t\t\tCurrent best Epoch: {best_epoch}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
        print(f"\t\t\tCurrent best Val ACC: {best_val_acc}")

        # Check for early stopping
        if epochs_no_improve == n_epochs_stop:
            print("\tEarly stopping!")
            break

        # Stop the time for the epoch
        end_time_epoch = time.time()
        hours_epoch, minutes_epoch, seconds_epoch = calculate_passed_time(start_time_epoch, end_time_epoch)
        print("\t\t\tElapsed time for epoch: {:0>2}:{:0>2}:{:05.2f}".format(hours_epoch, minutes_epoch, seconds_epoch))
    print(f"\tSave the trained model ...")
    Path(config['output_path']).mkdir(parents=True, exist_ok=True)
    save_name = f"{config['output_path']}/{config['dataset']}_{config['img_size']}_{config['training_procedure']}_{config['architecture']}_s{config['seed']}"
    torch.save(model.state_dict(), f"{save_name}_final.pth")
    torch.save(best_model.state_dict(), f"{save_name}_best.pth")

    print(f"\tFinished training.")
    # Stop the time
    end_time = time.time()
    hours, minutes, seconds = calculate_passed_time(start_time, end_time)
    print("\tElapsed time for training: {:0>2}:{:0>2}:{:05.2f}".format(hours, minutes, seconds))

    def plot_training_curves(history):
        epochs = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(12, 5))

        # --- Loss ---
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.plot(epochs, history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves")
        plt.legend()
        plt.grid(True)

        # --- Accuracy ---
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history["train_acc"], label="Train Acc")
        plt.plot(epochs, history["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curves")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        image_save_name = "learning_tracking_image"
        plt.savefig(config['output_path']+image_save_name+"_"+str(config['seed'])+".png")
    # Plot the training plots and save them
    print(f"Create the training plots and save them.")
    plot_training_curves(history)
