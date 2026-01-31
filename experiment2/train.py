"""
xAILab
Chair of Explainable Machine Learning
Otto-Friedrich University of Bamberg

@description:
The script trains a model on a specified dataset of the MedMNIST+ collection and saves the best performing model.
"""

# Import packages
import sys  #new
import torch

# Import packages
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

from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from timm.optim import AdamW
from torchvision.models import alexnet, AlexNet_Weights
from medmnist import INFO
import torch.optim as optim

# Import custom modules
# Assuming these exist in your project structure
from utils import calculate_passed_time, seed_worker, get_ACC, get_AUC

def train(config: dict, train_loader: DataLoader, val_loader: DataLoader):
    """
    Train a model on the specified dataset and save the best performing model.

    :param config: Dictionary containing the parameters and hyperparameters.
    :param train_loader: DataLoader for the training set.
    :param val_loader: DataLoader for the validation set.
    """

    #Start Code
    start_time = time.time()
    print("\tStart training ...")

    # Create the model
    print("\tCreate the model ...")
    if config['architecture'] == 'alexnet':
        model = alexnet(weights=AlexNet_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(4096, config['num_classes'])
    else:
        model = timm.create_model(config['architecture'], pretrained=True, num_classes=config['num_classes'])

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

    # Build optimizer param groups
    param_groups = []

    # Decoder always trained
    param_groups = [{"params": decoder_params, "lr": config["learning_rate_decoder"]}]

    # Encoder only trained in end-to-end
    if config['training_procedure'] == 'endToEnd':
        param_groups.append({"params": encoder_params, "lr": config["learning_rate_encoder"]})

    optimizer = optim.AdamW(param_groups)


    # Scheduler (must match number of param groups)
    decoder_lambda = lambda step: 0.9 ** (step // 200)
    encoder_lambda = lambda step: 1.0

    if config['training_procedure'] == 'endToEnd':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[decoder_lambda, encoder_lambda])
    else:
        # Only decoder param group exists
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[decoder_lambda])

    # Loss function + prediction fn
    print("\tDefine the loss function ...", config['task'])

    if config['task'] == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss().to(config['device'])
        prediction = nn.Sigmoid()
    else:
        criterion = nn.CrossEntropyLoss().to(config['device'])
        prediction = nn.Softmax(dim=1)

    # Best model tracking
    print("\tInitialize helper variables ...")

    best_val_loss, best_iter = np.inf, 0
    best_model = deepcopy(model).cpu()


    # Iteration-based setup
    max_iters = config['iterations']
    lr_decay_every = 200

    train_loss = 0.0
    epoch = 0
    epochs_without_improvement = 0
    #patience = 1000000

    train_iter = iter(train_loader)
    iters_per_epoch = len(train_loader)
    print(f"\tRun the training for {max_iters} iterations ...")

    for iteration in tqdm(range(max_iters)):
        print("current iteration: ", iteration)
        # Fetch batch (iteration-based)
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)
            train_loss = 0

        # Training step
        model.train()
        #torch.cuda.empty_cache() # <‑‑ ADD HERE
        images = images.to(config['device'])
        

        if config['task'] == 'multi-label, binary-class':
            labels = labels.to(torch.float32).to(config['device'])
        else:
            labels = torch.squeeze(labels, 1).long().to(config['device'])
        
        optimizer.zero_grad(set_to_none=True)
        # Run the forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Compute the loss and perform backpropagation
        train_loss += loss.item()
        loss.backward()

        # Update the weights        
        optimizer.step()
        

        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != old_lr:
            print(f"Decoder LR decayed to {new_lr:.6f} at iteration {iteration+1}")

        # Epoch boundary check (iteration-based)
        if (iteration + 1) % iters_per_epoch == 0:
            start_time_epoch = time.time()
            epoch += 1
            print(f"\n\tEpoch {epoch} finished -> Evaluate:")

            # ---------------- VALIDATION ----------------
            model.eval()
            val_loss = 0.0
            y_true_list = []
            y_pred_list = []

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validating", ncols=100):

                    images = images.to(config['device'])
                    outputs = model(images)

                    if config['task'] == 'multi-label, binary-class':
                        labels = labels.to(torch.float32).to(config['device'])
                        loss = criterion(outputs, labels)
                        outputs = prediction(outputs)

                    else:
                        labels = torch.squeeze(labels, 1).long().to(config['device'])
                        loss = criterion(outputs, labels)
                        outputs = prediction(outputs)
                        labels = labels.float().resize_(len(labels), 1)

                   # Store the current loss
                    val_loss += loss.item()
                    y_true_list.append(labels.cpu())
                    y_pred_list.append(outputs.cpu())

            y_true = torch.cat(y_true_list, dim=0)
            y_pred = torch.cat(y_pred_list, dim=0)
            # Aggregate metrics
            #y_true = torch.cat(y_true_list, dim=0)
            #y_logits = torch.cat(y_logits_list, dim=0)

            # Calculate the metrics
            val_acc = get_ACC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])
            val_auc = get_AUC(y_true.cpu().numpy(), y_pred.cpu().numpy(), config['task'])

            # Print the loss values and send them to wandb
            train_loss = train_loss / iters_per_epoch
            val_loss /= len(val_loader)

            print(f"\t\tTrain Loss: {train_loss}")
            print(f"\t\tVal   Loss: {val_loss}")
            print(f"\t\tACC:        {val_acc}")
            print(f"\t\tAUC:        {val_auc}")

            # Store the current best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iter = iteration
                best_model = deepcopy(model).cpu()
                epochs_without_improvement = 0 # Reset the counter
            else:
                epochs_without_improvement += 1 # Increment the counter
                print(f"\t\tEpochs without improvement: {epochs_without_improvement}")

            # Check for early stopping
            #if epochs_without_improvement >= patience:
            #    print(f"\nEarly stopping triggered after {patience} epochs without improvement.")
            #    break

            print(f"\t\tCurrent best Val Loss: {best_val_loss:.4f}")
            print(f"\t\tCurrent best Iter:     {best_iter}")

            # Reset epoch stats
            train_loss = 0.0

            # Stop the time for the epoch
            end_time_epoch = time.time()
            hours_epoch, minutes_epoch, seconds_epoch = calculate_passed_time(start_time_epoch, end_time_epoch)
            print("\t\t\tElapsed time for epoch: {:0>2}:{:0>2}:{:05.2f}".format(hours_epoch, minutes_epoch, seconds_epoch))

    
    # Save models
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