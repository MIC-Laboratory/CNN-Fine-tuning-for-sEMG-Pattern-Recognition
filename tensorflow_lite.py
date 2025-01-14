"""
    Description: Compile Tensorflow model to Tensorflow Lite model
    Author: Jimmy L. @ SF State MIC Lab
    Date: Summer 2022
"""

from model import *
import config
import numpy as np
import tensorflow as tf
from dataset import *
import random
import binascii

num_classes = 4
prev_params = [num_classes, config.filters, config.neurons, config.dropout, config.k_size, config.in_shape, config.p_kernel]



if __name__ == "__main__":
    # NOTE: Check if Utilizing GPU device
    print(tf.config.list_physical_devices('GPU'))
    
    # NOTE: Data Preprocessings
    
    # Get sEMG samples and labels. (shape: [num samples, 8(sensors/channels)])
    emg, label = folder_extract(
        config.folder_path,
        exercises=config.exercises,
        myo_pref=config.myo_pref
    )
    # Apply Standarization to data, save collected MEAN and STANDARD DEVIATION in the dataset to json
    emg = standarization(emg, config.std_mean_path)
    # Extract sEMG signals for wanted gestures.
    gest = gestures(emg, label, targets=config.targets)
    # Perform train test split
    train_gestures, test_gestures = train_test_split(gest)
    
    # NOTE: optional visualization that graphs class/gesture distributions
    # plot_distribution(train_gestures)
    # plot_distribution(test_gestures)
    
    # Convert sEMG data to signal images.
    X_train, y_train = apply_window(train_gestures, window=config.window, step=config.step)
    # Convert sEMG data to signal images.
    X_test, y_test = apply_window(test_gestures, window=config.window, step=config.step)
    
    # Convert data to correct input shape
    X_train = X_train.reshape(-1, 8, config.window, 1)
    X_test = X_test.reshape(-1, 8, config.window, 1)
    
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    
    print("Shape of Inputs:\n")
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    print("Data Type of Inputs\n")
    print(X_train.dtype)
    print(X_test.dtype)
    print("\n")
    
    # Shuffle data
    rand_arr1 = [i for i in range(24059)]
    rand_arr = [i for i in range(8012)]
    random.shuffle(rand_arr1)
    random.shuffle(rand_arr)
    
    new_X1 = []
    new_y1 = []
    for i, rand_i in enumerate(rand_arr1):
        new_X1.append(X_train[rand_i ])
        new_y1.append(y_train[rand_i ])
        
    new_X = []
    new_y = []
    for i, rand_i in enumerate(rand_arr):
        new_X.append(X_test[rand_i ])
        new_y.append(y_test[rand_i ])

    new_X1 = np.array(new_X1)
    new_y1 = np.array(new_y1)
    new_X = np.array(new_X)
    new_y = np.array(new_y)


    new_X1 = new_X1.astype(np.float32)
    new_X = new_X.astype(np.float32)
    
    # Load finetuned model finetuned/checkpoint.ckpt
    model = get_pretrained("finetuned/checkpoint.weights.h5", prev_params)
    
    # initialize object for Tensorflow Lite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Create representative dataset for TF Lite converter
    def representative_dataset_gen():
        for i, semg in enumerate(tf.data.Dataset.from_tensor_slices(new_X).batch(1).take(100)):
            semg = tf.cast(semg, tf.float32)
            yield [semg]

    # Load necessary metrics
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    
    # Begin Conversion
    tflite_model_quant = converter.convert()
    
    ## Show the quantized model size in KBs.
    tflite_model_size = len(tflite_model_quant) / 1024
    print('Quantized model size = %dKBs.' % tflite_model_size)
    # Save the model to disk
    open(config.tflite_path, "wb").write(tflite_model_quant)
    
    ## Output the quantized tflite model to a c-style header
    def convert_to_c_array(bytes) -> str:
        hexstr = binascii.hexlify(bytes).decode("UTF-8")
        hexstr = hexstr.upper()
        array = ["0x" + hexstr[i:i + 2] for i in range(0, len(hexstr), 2)]
        array = [array[i:i+10] for i in range(0, len(array), 10)]
        return ",\n  ".join([", ".join(e) for e in array])

    tflite_binary = open(config.tflite_path, 'rb').read()
    ascii_bytes = convert_to_c_array(tflite_binary)
    header_file = "const unsigned char model_tflite[] = {\n  " + ascii_bytes + "\n};\nunsigned int model_tflite_len = " + str(len(tflite_binary)) + ";"
    with open(config.tflite_c_path, "w") as f:
        f.write(header_file)