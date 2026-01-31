# tune.py
# run hyperparameter tuning with optuna

import sys
import time
import random
import yaml
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

import torchvision.transforms as transforms
from torchvision.models import alexnet, AlexNet_Weights

import timm
import medmnist
from medmnist import INFO
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour,
)

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import calculate_passed_time, seed_worker, get_ACC, get_AUC


class DualLogger:
    def __init__(self, logfile_path):
        self.terminal = sys.stdout
        self.log = open(logfile_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # Needed for Python's internal buffering
        self.terminal.flush()
        self.log.flush()


# Load config yaml file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Activate dual logging
save_name = f"{config['dataset']}_{config['img_size']}_{config['training_procedure']}_{config['architecture']}"

sys.stdout = DualLogger(config['output_path']+"/optuna_output_"+save_name+".log")


if config['architecture'] == 'alexnet':
    torch.backends.cudnn.benchmark = True  # Enable the benchmark mode in cudnn
    torch.backends.cudnn.deterministic = False  # Disable cudnn to be deterministic
    torch.use_deterministic_algorithms(False)  # Disable only deterministic algorithms

else:
    torch.backends.cudnn.benchmark = True  # Disable the benchmark mode in cudnn
    torch.backends.cudnn.deterministic = False  # Enable cudnn to be deterministic

    if config['architecture'] == 'samvit_base_patch16':
        torch.use_deterministic_algorithms(True, warn_only=True)

    else:
        torch.use_deterministic_algorithms(True)  # Enable only deterministic algorithms

#torch.manual_seed(config['seed'])  # Seed the pytorch RNG for all devices (both CPU and CUDA)
#random.seed(config['seed'])
#np.random.seed(config['seed'])
#g = torch.Generator()
#g.manual_seed(config['seed'])

# Extract the dataset and its metadata
info = INFO[config['dataset']]
config['task'], config['in_channel'], config['num_classes'] = info['task'], info['n_channels'], len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

# Create the data transforms and normalize with imagenet statistics
if config['architecture'] == 'alexnet':
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # Use ImageNet statistics
else:
    m = timm.create_model(config['architecture'], pretrained=True)
    mean, std = m.default_cfg['mean'], m.default_cfg['std']

# Training data gets augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(config['img_size'], scale=(0.8, 1.0)), # Zoom in/out
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

train_dataset = DataClass(split='train', transform=train_transform, download=False, as_rgb=True, size=config['img_size'], root=config['data_path'])
val_dataset = DataClass(split='val', transform=data_transform, download=False, as_rgb=True, size=config['img_size'], root=config['data_path'])

# Compute Class Imbalance (Weighted Loss)
targets = np.array(train_dataset.labels).squeeze().astype(int)
class_counts = np.bincount(targets)
num_classes = len(class_counts)
# Class weights for loss
# Inverse-frequency class weights (for sampler)
class_weights = torch.tensor( [1.0 / c for c in class_counts], dtype=torch.float )
sample_weights = torch.tensor( [class_weights[t] for t in targets], dtype=torch.float )

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True, )

train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, shuffle=False, num_workers=4)
train_loader_at_eval = DataLoader(train_dataset, batch_size=config['batch_size_eval'], shuffle=False, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

# The Objective Function
def objective(trial):
    # --- Hyperparameters to optimize ---
    lr_encoder = trial.suggest_float("lr_encoder", 1e-7, 1e-4, log=True)
    lr_decoder = trial.suggest_float("lr_decoder", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

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

    # Create the optimizer and the learning rate scheduler
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
            {"params": encoder_params, "lr": lr_encoder}, {"params": decoder_params, "lr": lr_decoder},
        ],
        weight_decay=weight_decay
    )

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
        eta_min=1e-7
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
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing) #CEL with Label Smoothing 
        prediction = nn.Softmax(dim=1)

    # Create variables to store the best performing model
    print("\tInitialize helper variables ...")
    best_val_loss, best_epoch = np.inf, 0
    val_acc = 0
    best_val_acc = 0.0
    epochs_no_improve = 0  # Counter for epochs without improvement
    n_epochs_stop = 50  # Number of epochs to wait before stopping

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
                #labels = labels.float().to(config['device'])
            else:
                labels = torch.squeeze(labels, 1).long().to(config['device'])
                #labels = labels.squeeze(1).long().to(config['device'])

            
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


            # TRAIN ACCURACY
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

        # Evaluation
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
        
        # --- Optuna Pruning ---
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if best_val_acc is None:
        best_val_acc = 0.0
    return best_val_acc

# Run Optimization
study = optuna.create_study(study_name="A", direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, 100)

print("\nBest Parameters Found:")
print(study.best_params)

# Now visualize
fig_optimization_history = plot_optimization_history(study)
fig_optimization_history.write_image(config['output_path']+"fig_optimization_history.png")
fig_param_importances = plot_param_importances(study)
fig_param_importances.write_image(config['output_path']+"fig_param_importances.png")

fig_parallel = plot_parallel_coordinate(study)
fig_parallel.write_image(config['output_path']+"fig_parallel_coordinate.png")

fig_slice = plot_slice(study)
fig_slice.write_image(config['output_path']+"fig_slice.png")

fig_contour = plot_contour(study)
fig_contour.write_image(config['output_path']+"fig_contour.png")

df = study.trials_dataframe()
df.to_csv("optuna_trials.csv", index=False)