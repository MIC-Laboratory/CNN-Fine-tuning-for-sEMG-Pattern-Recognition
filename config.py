"""
    Description: Configuration Variables and Parameters
    Author: Stella Parker @ SF State MIC Lab
    Date: Started: October 2024 -Ongoing
"""
# How many samples each sEMG image channel contains.
window = 32 #32

# Sliding step (for overlapping)
step = 16

# Kernel size for CNN
kernel_size = (3, 3) #(5, 3) #(3, 3 )

# Input shape for CNN
input_shape = (8, window, 1) #(8, window, 1) (8, 32, 1)

# Pool kernel for CNN maxpooling
pool_size =  (1, 2) #(1, 1) (3, 1)

# Exercises with dedicated gestures stored
exercises = ["E2"]

# Path of Ninapro DataBase 5 sEMG dataset.
folder_path = "Ninapro_DB5"

# Ninapro DB5 data collected via 2 Myo armband, controls which armband's 8 sensors to collect
myo_pref = "elbow"

# Class of gestures for training finetune-base model. Value indexes are based on "visuals/gestures.png"
targets = [0, 13, 15, 17, 18, 25, 26]
"""
relax, thumbs up, flexion, open hand, fist, wrist flexion, wrist extension
"""

# Path to save fintune-base model. (Make sure ends with '.ckpt').
save_path = "checkpoints/model.ckpt"

# Number of gestures to detect for finetune-base model.
num_classes = len(targets) # 7

# Number of CNN output filters the model contains.
filters = [48, 96]

# Number of neurons for FFN the model contains.
neurons = None #[512, 256]

# Whether to use depthwise seperatble CNN to reduce computation and parameters
seperable_cnn = False

# Dropout rate.
dropout = 0.5

# The number of epochs without improvement after which training will be early stopped
patience = 50

# Initial learning rate for training finetune-base model.
inital_lr = 0.2

# Number of training epochs for the finetune-base model.
epochs = 200

# Batch size for training the finetune-base model.
batch_size = 384

# Paths for saving logs generated when training finetune-base model.
acc_log = 'visuals/acc_log.jpg'
loss_log = 'visuals/loss_log.jpg'

# Params info needed to load pretrained finetune-base model.
prev_params = [num_classes, filters, neurons, dropout, kernel_size, input_shape, pool_size]

# Path of json with MEAN and Standard Deviation for each sensor Channel.
std_mean_path = "scaling_params.json"

tflite_path = 'tensorflow_lite/model.tflite'

tflite_c_path = "tensorflow_lite/model.h"
