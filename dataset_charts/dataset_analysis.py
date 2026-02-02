import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from medmnist import INFO
import medmnist
import torch
import os
import pandas as pd
from tabulate import tabulate

mnist_data_path = "/mnt/data/user/data"
base_path = "/mnt/data/user/Charts_Dataset"
save_images = True

# Count occurrences per class
def count_classes(dataset, n_classes, multilabel=False):
    counts = np.zeros(n_classes, dtype=int)
    for _, label in dataset:
        if multilabel:
            label = label.numpy() if hasattr(label, "numpy") else label
            counts += label.astype(int)
        else:
            idx = int(label) if np.isscalar(label) else int(label[0])
            counts[idx] += 1
    return counts

# Plot sample image per class
def display_class_samples(dataset, df, n_classes, multilabel=False):
    seen = [False] * n_classes
    samples = [None] * n_classes

    for img, label in dataset:
        label_arr = label.numpy() if hasattr(label, "numpy") else label
        if multilabel:
            for class_idx, active in enumerate(label_arr):
                if active and not seen[class_idx]:
                    samples[class_idx] = (img, class_idx)
                    seen[class_idx] = True
                    break
        else:
            class_idx = int(label) if np.isscalar(label) else int(label[0])
            if not seen[class_idx]:
                samples[class_idx] = (img, class_idx)
                seen[class_idx] = True
        if all(seen):
            break

    fig, axes = plt.subplots(1, n_classes, figsize=(n_classes * 2, 3))
    if n_classes == 1:
        axes = [axes]

    for ax, sample in zip(axes, samples):
        if sample is None:
            ax.axis('off')
            continue
        img, class_idx = sample
        img_np = np.array(img)
        cmap = 'gray' if img.mode == 'L' else None
        ax.imshow(img_np, cmap=cmap)
        ax.set_title(f"{class_idx}", fontsize=9, pad=2)
        short_label = df.loc[df["Class"] == class_idx, "Short Label"].values[0]
        ax.text(0.5, -0.15, short_label, fontsize=9, ha='center', va='top', transform=ax.transAxes)
        ax.axis('off')
    fig.suptitle(f"{dataset_name.upper()} Images", fontsize=14, y=1.0)
    plt.tight_layout()
    if save_images:
        plt.savefig(full_path + "/" + str(dataset_name.capitalize()) + "_Images.png", dpi=300, bbox_inches='tight')
    plt.show()

def render_table_as_image(df, title, full_path):
    fig, ax = plt.subplots(figsize=(min(20, len(df.columns) * 2), 0.5 + 0.4 * len(df)))

    # Hide axes
    ax.axis('off')

    # Create the table
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        colLoc='center'
    )

    # Style it a bit
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # Optional: Add title
    if title:
        plt.title(title, fontsize=14, pad=10)

    plt.tight_layout()

    # Save or show
    if save_images:
        plt.savefig(full_path + "/" + str(dataset_name.capitalize()) + "_Data_Table.png", dpi=300, bbox_inches='tight')
    
    plt.show()

# Loop through datasets
for dataset_name in ('chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist',
    'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist',
    'organamnist', 'organcmnist', 'organsmnist'):

    full_path = os.path.join(base_path, dataset_name)
    if save_images:
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            print(f" Created folder: {full_path}")
        else:
            print(f" Folder already exists: {full_path}")

    print(f"\n Dataset: {dataset_name.upper()}")

    info = INFO[dataset_name]
    label_names = info['label']
    n_classes = len(label_names)
    multilabel = dataset_name == 'chestmnist'

    DataClass = getattr(medmnist, info['python_class'])
    train = DataClass(split='train', download=False, size = 224, root=mnist_data_path)
    val = DataClass(split='val', download=False, size = 224, root=mnist_data_path)
    test = DataClass(split='test', download=False, size = 224, root=mnist_data_path)

    train_counts = count_classes(train, n_classes, multilabel)
    val_counts = count_classes(val, n_classes, multilabel)
    test_counts = count_classes(test, n_classes, multilabel)
    total_counts = train_counts + val_counts + test_counts

    train_total = train_counts.sum()
    val_total = val_counts.sum()
    test_total = test_counts.sum()
    total_total = total_counts.sum()

    retina_labels = {
        "0": "No DR", "1": "Mild NPDR", "2": "Moderate NPDR",
        "3": "Severe NPDR", "4": "Proliferative DR"
    }

    label_map = {
        "colorectal adenocarcinoma epithelium": "colorectal adenocarcinoma",
        "actinic keratoses and intraepithelial carcinoma": "actinic keratoses",
        "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)": "immature granulocytes"
    }

    # Build table with %-annotated values and short labels
    table = []
    for idx in range(n_classes):

        raw_label = label_names.get(str(idx), f"Class {idx}")
        if "retina" in dataset_name:
            raw_label = retina_labels.get(str(idx), f"Class {idx}")
        short_label = label_map.get(raw_label, raw_label)

        train_str = f"{train_counts[idx]} ({train_counts[idx]/train_total:.1%})"
        val_str   = f"{val_counts[idx]} ({val_counts[idx]/val_total:.1%})"
        test_str  = f"{test_counts[idx]} ({test_counts[idx]/test_total:.1%})"
        total_str = f"{total_counts[idx]} ({total_counts[idx]/total_total:.1%})"

        table.append([idx, raw_label, train_str, val_str, test_str, total_str, short_label])

    headers = ["Class", "Label", "Train", "Val", "Test", "Total", "Short Label"]
    

    # Bar chart of counts
    x = np.arange(n_classes)
    # Add total row
    train_sum = sum(train_counts)
    val_sum = sum(val_counts)
    test_sum = sum(test_counts)
    total_sum = sum(total_counts)

    train_str_total = f"{train_sum} (100.0%)"
    val_str_total   = f"{val_sum} (100.0%)"
    test_str_total  = f"{test_sum} (100.0%)"
    total_str_total = f"{total_sum} (100.0%)"

    table.append(["Total", "", train_str_total, val_str_total, test_str_total, total_str_total, ""])
    df = pd.DataFrame(table, columns=headers)
    print(tabulate(df, headers=headers, tablefmt="github"))
    # Drop the "Label" and "Short Label" columns
    df_render = df.drop(columns=["Label", "Short Label"])

    # Call the image rendering function
    render_table_as_image(df_render, f"{dataset_name.upper()} Dataset Class Distribution", full_path) # might add save-path


    # Show one image per class
    display_class_samples(train, df, n_classes, multilabel)
    width = 0.18
    gap = 0.02

    color1 = "#1C2E40"  # Total
    color2 = "#396e9d"  # Train
    color3 = "#6ca6cd"  # Val
    color4 = "#add8e6"  # Test

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5*(width + gap), total_counts, width=width, label='Total',      color=color1)
    ax.bar(x - 0.5*(width + gap), train_counts, width=width, label='Train',      color=color2)
    ax.bar(x + 0.5*(width + gap), val_counts,   width=width, label='Validation', color=color3)
    ax.bar(x + 1.5*(width + gap), test_counts,  width=width, label='Test',       color=color4)

    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_classes)])
    ax.set_ylabel("Sample Count")
    ax.set_xlabel("Class")
    fig.suptitle(f"{dataset_name.upper()} Distribution: Total / Train / Val / Test", fontsize=14, y=1.02)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    if save_images:
        plt.savefig(full_path + "/" + str(dataset_name.capitalize()) + "_Class_Distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
