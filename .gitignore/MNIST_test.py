##############################################
# train_mnist_subset.py
##############################################
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import config
from model import get_model, train_model

def load_mnist_subset_4classes():
    """
    Downloads MNIST, resizes to (8,32) if needed,
    and KEEPS ONLY DIGITS [0,1,2,3].
    
    Returns X_train, y_train, X_test, y_test
    in shape (N,1,8,32) for X.
    """

    # Define transform to resize MNIST to (8,32) 
    #    because model expects (1,8,32).
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((8, 32)),  # same as your sEMG shape
        transforms.ToTensor(),       # => (1,8,32)
    ])

    #  Download full MNIST (digits 0..9)
    train_full = torchvision.datasets.MNIST(
        root="./mnist_data",
        train=True,
        download=True,
        transform=transform
    )
    test_full = torchvision.datasets.MNIST(
        root="./mnist_data",
        train=False,
        download=True,
        transform=transform
    )

    # Filter to keep only digits 0..3
    train_data = []
    for img_tensor, label in train_full:
        if label in [0, 1, 2, 3]:
            train_data.append((img_tensor, label))

    test_data = []
    for img_tensor, label in test_full:
        if label in [0, 1, 2, 3]:
            test_data.append((img_tensor, label))

    # Convert to NumPy arrays
    X_train_list, y_train_list = [], []
    for (img_tensor, label) in train_data:
        X_train_list.append(img_tensor.numpy())  # shape (1,8,32)
        y_train_list.append(label)

    X_test_list, y_test_list = [], []
    for (img_tensor, label) in test_data:
        X_test_list.append(img_tensor.numpy())
        y_test_list.append(label)

    X_train = np.array(X_train_list, dtype=np.float32)  # => (N,1,8,32)
    y_train = np.array(y_train_list, dtype=np.int64)
    X_test  = np.array(X_test_list,  dtype=np.float32)
    y_test  = np.array(y_test_list,  dtype=np.int64)

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    #Check GPU
    gpu_available = torch.cuda.is_available()
    print(f"GPU available? {gpu_available}\n")

    #  Load the 4-class subset of MNIST (digits 0..3)
    X_train, y_train, X_test, y_test = load_mnist_subset_4classes()

    print("Subset MNIST shapes (only digits [0..3]):")
    print("  X_train:", X_train.shape)
    print("  y_train:", y_train.shape)
    print("  X_test: ", X_test.shape)
    print("  y_test: ", y_test.shape)
    print()

    #  Build  model, which is locked to num_classes=4
    cnn = get_model(
        num_classes=config.num_classes,  #  4
        filters=config.filters,
        neurons=config.neurons,
        dropout=config.dropout,
        kernel_size=config.kernel_size,
        input_shape=config.input_shape,
        pool_size=config.pool_size
    )

    # Train using existing train_model
    history = train_model(
        model=cnn,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        batch_size=config.batch_size,
        save_path=config.save_path,
        epochs=config.epochs,
        patience=config.patience,
        lr=config.inital_lr
    )

    print("\nSubset MNIST (digits 0..3) training complete!")

