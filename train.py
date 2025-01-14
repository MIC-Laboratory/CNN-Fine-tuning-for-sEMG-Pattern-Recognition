"""
    Description:
        Achieves:
            - Data Preprocessing over Ninapro DataBase5
            - Training finetune-base model (Saving weights along the way)
            - Visualize training logs (model accuracy and loss during training)
            
    Author: Stella Parker @ SF State MIC Lab
    Date: Started: October 2024 -Ongoing
"""
import torch
import numpy as np
import config
from dataset import folder_extract, standarization,gestures,train_test_split, apply_window
from model import (get_model, train_model, plot_logs, get_finetune   # If you plan to finetune
    # create_finetune, # If you plan to create a finetune model
)

if __name__ == "__main__":
    # 1. Check GPU availability (PyTorch style)
    gpu_available = torch.cuda.is_available()
    print(f"GPU available? {gpu_available}")

    # Data Preprocessing (uses dataset.py)
    # Extract sEMG signals and labels
    emg, label = folder_extract(
        config.folder_path,
        exercises=config.exercises,
        myo_pref=config.myo_pref
    )

    # Standardize signals (and optionally save mean/std to JSON)
    emg = standarization(emg, config.std_mean_path)

    # Keep only certain gesture labels
    gest = gestures(emg, label, targets=config.targets)

    # Train-test split
    train_gestures, test_gestures = train_test_split(gest)
    
    # (Optionally visualize gesture distribution)
    # plot_distribution(train_gestures)
    # plot_distribution(test_gestures)

    # Convert sEMG data to image windows
    X_train, y_train = apply_window(train_gestures, window=config.window, step=config.step)
    X_test, y_test = apply_window(test_gestures, window=config.window, step=config.step)

    # Reshape for CNN
    X_train = X_train.reshape(-1, 8, config.window, 1)
    X_test  = X_test.reshape(-1, 8, config.window, 1)

    X_train = np.transpose(X_train, (0, 3, 1, 2))  # => (N, 1, 8, 32)
    X_test  = np.transpose(X_test, (0, 3, 1, 2))

    # Convert to float32
    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    print("Shape of Inputs:")
    print("  X_train:", X_train.shape)
    print("  y_train:", y_train.shape)
    print("  X_test: ", X_test.shape)
    print("  y_test: ", y_test.shape)
    print("Data Types:")
    print("  X_train dtype:", X_train.dtype)
    print("  X_test dtype: ", X_test.dtype)
    print()

    # get_model() will return the new model
    cnn = get_model(
        num_classes=config.num_classes,
        filters=config.filters,
        neurons=config.neurons,
        dropout=config.dropout,
        kernel_size=config.kernel_size,
        input_shape=config.input_shape, 
        pool_size=config.pool_size
    )

    # train_model() will return a history object that contains training logs
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
    print("Training complete!")
    
    # done_msg = "Training complete! Model saved successfully."
    # print(done_msg)
    # with open(config.log_path, "a") as f:
    # f.write(done_msg + "\n")
    
    # # If you have stats from `history` you want to print:
    # final_loss = history["val_loss"][-1] if len(history["val_loss"]) > 0 else None
    # final_acc  = history["val_accuracy"][-1] if len(history["val_accuracy"]) > 0 else None
    # summary_msg = f"Final Validation Loss: {final_loss}, Final Validation Accuracy: {final_acc}"
    # print(summary_msg)
    # f.write(summary_msg + "\n")
    # # NOTE: Optional test for loaded model's performance
    # model.compile(
    #         optimizer=tf.keras.optimizers.Adam(learning_rate=0.2),
    #         loss='sparse_categorical_crossentropy',
    #         metrics=['accuracy'],
    #     )
    # # See if weights were the same
    # model.evaluate(X_test, y_test)
    
    # # # Test with finetune model. (last classifier block removed from base model)
    # # finetune_model = get_finetune(config.save_path, config.prev_params, num_classes=config.num_classes)
    # # print("finetune model loaded!")
    
    # # NOTE: You can load finetune model like this too.
    # finetune_model = create_finetune(model, num_classes=4)
    # finetune_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.2),
    #     loss='sparse_categorical_crossentropy',
    #     metrics=['accuracy'],
    # )
    # finetune_model.evaluate(X_test, y_test)