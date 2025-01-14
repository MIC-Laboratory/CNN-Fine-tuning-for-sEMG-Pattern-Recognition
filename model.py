"""
Description: Code for A.I. model implementation and utility functions.
Author: Stella Parker @ SF State MIC Lab
Date: Started: October 2024 -Ongoing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import copy


def get_model(
    num_classes=4,
    filters=None,
    neurons=None,
    dropout=0.5,
    kernel_size=(5, 3),
    input_shape=(52, 8, 1),
    pool_size=(3, 1),
):
    """
    Purpose:
        Establish the architecture for the finetune-base A.I. model.

    Args:
        num_classes (int, optional): Number of classes/gestures to classify. Defaults to 4.
        filters (list, optional): Output filters for the first and second 2D CNN. Defaults to None.
        neurons (list, optional): Number of neurons for the first and second neural network. Defaults to None.
        dropout (float, optional): Dropout rate. Defaults to 0.5.
        kernel_size (tuple, optional): Kernel window size for CNN. Defaults to (5, 3).
        input_shape (tuple, optional): Input shape for CNN. Defaults to (52, 8, 1).
        pool_size (tuple, optional): Max pool size. Defaults to (3, 1).

    Returns:
        model (nn.Module): The finetune-base model.
    """

    if filters is None:
        filters = [32, 64]

    class FinetuneBaseModel(nn.Module):
        def __init__(self, num_classes, filters, neurons, dropout, kernel_size, pool_size):
            super(FinetuneBaseModel, self).__init__()
            self.cnn1 = nn.Conv2d(
                in_channels=1,
                out_channels=filters[0],
                kernel_size=kernel_size, #kernel_size, keep track of this
                stride=1,
            )
            self.bn1 = nn.BatchNorm2d(filters[0])
            self.prelu1 = nn.PReLU()
            self.drop1 = nn.Dropout(p=dropout)
            self.pool1 = nn.MaxPool2d(pool_size)

            self.cnn2 = nn.Conv2d(
                in_channels=filters[0],
                out_channels=filters[1],
                kernel_size=kernel_size,
                stride=1,
            )
            self.bn2 = nn.BatchNorm2d(filters[1])
            self.prelu2 = nn.PReLU()
            self.drop2 = nn.Dropout(p=dropout)
            self.pool2 = nn.MaxPool2d(pool_size)

            self.neurons = neurons
            self.flatten = nn.Flatten()

            if (neurons is not None) and (len(neurons) > 0):
                ffn_modules = []
                first_layer = nn.LazyLinear(neurons[0], bias=True)
                ffn_modules.append(first_layer)
                ffn_modules.append(nn.PReLU())

                for i in range(len(neurons) - 1):
                    ffn_modules.append(nn.LazyLinear(neurons[i], neurons[i + 1]))
                    ffn_modules.append(nn.PReLU())

                self.ffn = nn.Sequential(*ffn_modules)
                self.classifier = nn.LazyLinear(neurons[-1], num_classes)
            else:
                self.ffn = None
                self.classifier = nn.LazyLinear(num_classes, bias=True)

            self.softmax = nn.Softmax(dim=1)

        def forward(self, x):
            x = self.cnn1(x)
            x = self.bn1(x)
            x = self.prelu1(x)
            x = self.drop1(x)
            x = self.pool1(x)

            x = self.cnn2(x)
            x = self.bn2(x)
            x = self.prelu2(x)
            x = self.drop2(x)
            x = self.pool2(x)

            x = self.flatten(x)

            if self.ffn is not None:
                x = self.ffn(x)
            x = self.classifier(x)
            x = self.softmax(x)
            return x

    model = FinetuneBaseModel(num_classes, filters, neurons, dropout, kernel_size, pool_size)
    return model


def create_finetune(model, num_classes=4):
    """
    Purpose:
        Generate a new finetune model from the pretrained base model.
        Creating a deep copy of the base_model so that the newly created
        finetune_model can be modified without altering the original base_model.

    Args:
        model (nn.Module): The pretrained finetune-base model.
        num_classes (int, optional): Number of gestures/classes for the new model. Defaults to 4.

    Returns:
        new_model (nn.Module): The new finetune model.
    """
    new_model = copy.deepcopy(model)
    if hasattr(new_model, "classifier"):
        old_in = new_model.classifier.in_features
        new_model.classifier = nn.LazyLinear(old_in, num_classes)
    new_model.softmax = nn.Softmax(dim=1)
    return new_model


def get_pretrained(path, prev_params):
    """
    Purpose:
        Load a pretrained finetune-base model given its checkpoint path.
        prev_params structure:
        prev_params[0] ⇒ num_classes
        prev_params[1] ⇒ filters ([32, 64])
        prev_params[2] ⇒ neurons ([512, 128] or None)
        prev_params[3] ⇒ dropout (0.5)
        prev_params[4] ⇒ kernel_size (5, 3))
        prev_params[5] ⇒ input_shape ((52, 8, 1))
        prev_params[6] ⇒ pool_size ((3, 1))

    Args:
        path (str): Path of pretrained weights of the finetune-base model.
        prev_params (list): Parameter specification of the pretrained finetune-base model.

    Returns:
        base_model (nn.Module): A PyTorch model loaded with the weights from 'path'.
    """
    base_model = get_model(
        num_classes=prev_params[0],
        filters=prev_params[1],
        neurons=prev_params[2],
        dropout=prev_params[3],
        kernel_size=prev_params[4],
        input_shape=prev_params[5],
        pool_size=prev_params[6],
    )
    base_model.load_state_dict(torch.load(path, map_location="gpu"))
    return base_model


def get_finetune(path, prev_params, lr=0.2, num_classes=4):
    """
    Purpose:
        Load a pretrained finetune model given its checkpoint path.
        prev_params structure:
        prev_params[0] ⇒ num_classes
        prev_params[1] ⇒ filters ([32, 64])
        prev_params[2] ⇒ neurons ([512, 128] or None)
        prev_params[3] ⇒ dropout (0.5)
        prev_params[4] ⇒ kernel_size (5, 3))
        prev_params[5] ⇒ input_shape ((52, 8, 1))
        prev_params[6] ⇒ pool_size ((3, 1))

    Args:
        path (str): Path of pretrained weights of the finetune model.
        prev_params (list): Parameter specification of the pretrained finetune model
        num_classes (int, optional): Number of gestures/classes for the new model. Defaults to 4.

    Returns:
        finetune_model (nn.Module): A PyTorch model loaded with the weights from 'path'.
    """
    base_model = get_model(
        num_classes=prev_params[0],
        filters=prev_params[1],
        neurons=prev_params[2],
        dropout=prev_params[3],
        kernel_size=prev_params[4],
        input_shape=prev_params[5],
        pool_size=prev_params[6],
    )
    base_model.load_state_dict(torch.load(path, map_location="gpu"))
    finetune_model = create_finetune(base_model, num_classes=num_classes)
    return finetune_model


def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    batch_size,
    save_path=None,
    epochs=200,
    patience=50,
    lr=0.2,
    decay_rate=0.9,
):
    """
    Purpose:
        Train the finetune-base model.

    Args:
        model (nn.Module): The finetune-base model to train.
        X_train (numpy.ndarray): The training input. Shape: [n_samples, 1, 8, 52].
        y_train (numpy.ndarray): The training target/label.
        X_test (numpy.ndarray): The testing input. Shape: [n_samples, 1, 8, 52].
        y_test (numpy.ndarray): The testing target/label.
        batch_size (int): Batch_size for training.
        save_path (str): Path to save the model's weights. Ends with '.pth' or '.ckpt'.
        epochs (int, optional): Number of training epochs. Defaults to 200.
        patience (int, optional): Number of epochs without improvement for early stopping. Defaults to 50. 80?
        lr (float, optional): Initial learning rate. Defaults to 0.2.
        decay_rate (float, optional): Exponential decay rate for LR. Defaults to 0.9.

    Returns:
        history (dict): A dictionary containing training and validation logs.
    """
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

        # scheduler.step()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_x.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}"
            f" | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}"
            f" | Val Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            if save_path is not None:
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    return history


def plot_logs(history, acc=True, save_path=None):
    """
    Purpose:
        Plot loss and accuracy logs from model training.

    Args:
        history (dict): The loss and accuracy log output from model training with
                        keys: ['loss', 'val_loss', 'accuracy', 'val_accuracy'].
        acc (bool, optional): Whether to plot training accuracy logs. Defaults to True.
        save_path (str, optional): Path to save plot. Should end with '.jpg'. Defaults to None.
    """
    if acc:
        params = {"acuracy", "val_accuracy", "model accuracy", "accuracy"}
    else:
        params = {"loss", "val_loss", "model loss", "loss"}

    plt.figure(figsize=(20, 6))
    plt.plot(history[params[0]], label="Train")
    plt.plot(history[params[1]], label="Validation")
    plt.title(params[2])
    plt.ylabel(params[3])
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def realtime_pred(model, sEMG, num_channels=8, window_length=32):
    """
    Purpose:
        Perform realtime predictions with the finetuned model.

    Args:
        model (nn.Module): The finetuned model.
        sEMG (numpy.ndarray): The realtime sEMG samples to input.
        num_channels (int, optional): Number of sensors/channels. Defaults to 8.
        window_length (int, optional): Samples included per sensor/channel. Defaults to 32.

    Returns:
        int: The model prediction index.
    """
    sEMG = np.array(sEMG).reshape(-1, 1, num_channels, window_length)
    device = next(model.parameters()).device
    sEMG_t = torch.tensor(sEMG, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        pred = model(sEMG_t)
        pred_idx = torch.argmax(pred, dim=1).item()

    return pred_idx
