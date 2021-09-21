import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import re

def main():

    logs = [
        # (log file, ending epoch)
        ("output/experiment/logs/training_0.log", 30),
        ("output/experiment/logs/training_30.log", 40),
        ("output/experiment/logs/training_40.log", 50),
    ]

    train_rank1, train_rank5, train_loss = [], [], []
    valid_rank1, valid_rank5, valid_loss = [], [], []

    for i, (log_file, end_epoch) in enumerate(logs):

        # Parse log file
        rows = open(log_file).read().strip()

        batch_train_rank1, batch_train_rank5, batch_train_loss = [], [], []
        # batch_valid_rank1, batch_valid_rank5, batch_valid_loss = [], [], []
        epochs = set(re.findall(r"Epoch\[(\d+)\]", rows))
        epochs = sorted([int(epoch) for epoch in epochs])

        # Extract rank1, rank5 and loss for training
        for epoch in epochs:
            rank1 = re.findall(rf"Epoch\[{str(epoch)}\] Batch.*accuracy=([0]*\.?[0-9]+)", rows)[-1]
            rank5 = re.findall(rf"Epoch\[{str(epoch)}\] Batch.*top_k_accuracy_5=([0]*\.?[0-9]+)", rows)[-1]
            loss = re.findall(rf"Epoch\[{str(epoch)}\] Batch.*cross-entropy=([0-9]\.?[0-9]+)", rows)[-1]
            batch_train_rank1.append(float(rank1))
            batch_train_rank5.append(float(rank5))
            batch_train_loss.append(float(loss))

        # Extract rank1, rank5, and loss for validation
        batch_valid_rank1 = re.findall(r"Validation-accuracy=(.*)", rows)
        batch_valid_rank5 = re.findall(r"Validation-top_k_accuracy_5=(.*)", rows)
        batch_valid_loss = re.findall(r"Validation-cross-entropy=(.*)", rows)
        batch_valid_rank1 = [float(rank1) for rank1 in batch_valid_rank1]
        batch_valid_rank5 = [float(rank5) for rank5 in batch_valid_rank5]
        batch_valid_loss = [float(loss) for loss in batch_valid_loss]

        end = end_epoch if i == 0 else logs[i-1][1]
        train_rank1.extend(batch_train_rank1[:end])
        train_rank5.extend(batch_train_rank5[:end])
        train_loss.extend(batch_train_loss[:end])
        valid_rank1.extend(batch_valid_rank1[:end])
        valid_rank5.extend(batch_valid_rank5[:end])
        valid_loss.extend(batch_valid_loss[:end])

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(train_rank1)), train_rank1, label="Train Rank1")
    plt.plot(np.arange(0, len(train_rank5)), train_rank5, label="Train Rank5")
    plt.plot(np.arange(0, len(valid_rank1)), valid_rank1, label="Valid Rank1")
    plt.plot(np.arange(0, len(valid_rank5)), valid_rank5, label="Valid Rank5")
    plt.title("ResNet: Rank1 / Rank5 Accuracy on ImageNet")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.savefig(os.path.sep.join(["output", "accuracy.png"]))
    #
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, len(train_loss)), train_loss, label="Train Loss")
    plt.plot(np.arange(0, len(valid_loss)), valid_loss, label="Valid Loss")
    plt.title("ResNet: Loss on ImageNet")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.savefig(os.path.sep.join(["output", "loss.png"]))

if __name__ == '__main__':
    main()
