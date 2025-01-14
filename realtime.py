## ##
## @original author: Amir Modan
## @editor: Jimmy L. @ SF State MIC Lab
##  - Date: Summer 2022

## Main Program for Real-Time system which establishes BLE connection,
##     defines GUI, and finetunes realtime samples from a pretrained finetune-base model.
    
## Flow: (running via Async Functions)
##     1. Run the code (python realtime.py)
##     2. Enable bluetooth in setting, and code with automatically pair with armband
##     3. Follow instructions to perform gestures for finetuning
##     4. Finetune training starts
##         - Optional: save finetuned-model
##     5. Real time gesture recognition begins

## Note:
##     1. Should see myo armband in blue lighting if connected.
##     2. Run this on Linux if possible, sometimes Bleak refuses to connect to Myo Armband under Windows environment.
## ##

import asyncio
import json
# import random as rand
import config
import nest_asyncio
nest_asyncio.apply()
# import tensorflow as tf
import torch as pytorch
import numpy as np
import warnings
from typing import Any
from bleak import BleakClient, discover
from dataset import realtime_preprocessing
from model import get_finetune, realtime_pred


warnings.filterwarnings("ignore")
tf.get_logger().setLevel('INFO')

## UUID's for BLE Connection

CONTROL = "d5060401-a904-deb9-4748-2c7f4a124842"
EMG0 = "d5060105-a904-deb9-4748-2c7f4a124842"
EMG1 = "d5060205-a904-deb9-4748-2c7f4a124842"
EMG2 = "d5060305-a904-deb9-4748-2c7f4a124842"
EMG3 = "d5060405-a904-deb9-4748-2c7f4a124842"

## Batch size for realtime fine-tuning
realtime_batch_size = 2

## Epoch for realtime fine-tuning
realtime_epochs = 15

## Samples window
window = 32

## Step Size
step_size = 10

## Samples to be recored for each gesture
SAMPLES_PER_GESTURE = 10 * window

## List of Gestures to be used for classification

GESTURES = [
    "Rest", "Fist", "Thumbs Up", "Ok Sign"
]

delay = 2.0

## Number of sensors Myo Armband contains
num_sensors = 8

## Path to save finetuned model, set NONE if no export
## finetuned_path = None
finetuned_path = "finetuned/checkpoint.ckpt"

## 2D list to store realtime training data
sensors = [[] for i in range(num_sensors)]

## Bluetooth device for Myo Armband
selected_device = []

## Load MEAN and Standard Deviation for Standarization from Ninapro DB5 sEMG signals.
with open(config.std_mean_path, 'r') as f:
    params = json.load(f)

    
class Connection:
    
    client: BleakClient = None
    
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        EMG0: str,
        EMG1: str,
        EMG2: str,
        EMG3: str,
        CONTROL: str,
    ):
        self.loop = loop
        self.EMG0 = EMG0 ## MyoCharacteristic0
        self.EMG1 = EMG1 ## MyoCharacteristic1
        self.EMG2 = EMG2 ## MyoCharacteristic2
        self.EMG3 = EMG3 ## MyoCharacteristic3
        self.CONTROL = CONTROL
        self.connected = False
        self.connected_device = None
        self.model = get_finetune(config.save_path, config.prev_params, lr=0.0002, num_classes=len(GESTURES))
        self.current_sample = [[] for i in range(num_sensors)]
        
        self.count = 0

    ## ##
    ##     Handler for when BLE device is disconnected
    
    ## ##
    def on_disconnect(self, client: BleakClient):
        self.connected = False
        print(f"Disconnected from {self.connected_device.name}!")

    # ##
    #     Callback right after BLE device is deisconnected
    
    # ##
    async def cleanup(self):
        ## Terminates all communication attempts with BLE device
        if self.client:
            await self.client.stop_notify(EMG0)
            await self.client.stop_notify(EMG1)
            await self.client.stop_notify(EMG2)
            await self.client.stop_notify(EMG3)
            await self.client.disconnect()

    # ##
    #     Searches for nearby BLE devices or initiates connection with BLE device if chosen
    # ##
    
    async def manager(self):
        print("Starting connection manager.")
        while True:
            if self.client:
                await self.connect()
            else:
                await self.select_device()
                await asyncio.sleep(1.0, loop=loop)       
    
    # ##
    #     Performs initial actions on connection with BLE device, including training neural network
    
    # ##
    async def connect(self):
        if self.connected:
            return
        try: 
            ## Stopped Here
            await asyncio.sleep(0.01, loop=loop)
            await self.client.connect()

            self.connected = await self.client.is_connected()
            if self.connected:
                print(F"Connected to {self.connected_device.name}")
                self.client.set_disconnected_callback(self.on_disconnect)
                
                ## Must send below command to Myo Armband to initiate EMG communication
                bytes_to_send = bytearray([1, 3, 2, 0, 0])
                await connection.client.write_gatt_char(CONTROL, bytes_to_send)
                
                ## Loops through each gesture and collects training data
                for gesture in GESTURES:
                    
                    ## Notify User to begin performing current gesture
                    print("Begining to Perform " + gesture)
                    
                    initial_length = len(sensors[0])
                    
                    ## Generate slight delay to allow time for user to perform next gesture
                    await asyncio.sleep(delay, loop=loop)
                    
                    ## Notify User Myo Armband is currently collecting signals
                    print("Perform " + gesture + " Now!\n")
                    
                    ## Begin collecting training data
                    await self.client.start_notify(self.EMG0, self.training_handler)
                    await self.client.start_notify(self.EMG1, self.training_handler)
                    await self.client.start_notify(self.EMG2, self.training_handler)
                    await self.client.start_notify(self.EMG3, self.training_handler)
                    
                    ## Continue until enough data is collected
                    while((len(sensors[0])-initial_length) < SAMPLES_PER_GESTURE):
                        await asyncio.sleep(0.05, loop=loop)
                        
                    ## Stop collecting training data
                    await self.client.stop_notify(EMG0)
                    await self.client.stop_notify(EMG1)
                    await self.client.stop_notify(EMG2)
                    await self.client.stop_notify(EMG3)
                    
                    ## If some channels sent more data than others,
                    ## discards extra data so all channels have the same amount of data
                    for channel_idx, sensor_samples in enumerate(sensors):
                        sensors[channel_idx] = sensor_samples[:(SAMPLES_PER_GESTURE+initial_length)]
                
                ## Get preprocessed data for training
                inputs, outputs = realtime_preprocessing(sensors, params_path=config.std_mean_path,
                                                         num_classes=len(GESTURES), window=window, step=step_size)
                
                ## Perform cross-validation sampling ([label 0, label 1, label 2, label 3, label 0, label 1, label 2, label 3, ...])
                sampled_inputs = []
                sampled_outputs = []

                clusters = len(inputs) // len(GESTURES)
                for idx in range(clusters):
                    for labels in range(len(GESTURES)):
                        sampled_inputs.append(inputs[idx+(labels*clusters)])
                        sampled_outputs.append(outputs[idx+(labels*clusters)])

                sampled_inputs = np.array(sampled_inputs)
                sampled_outputs = np.array(sampled_outputs)
                
                ## Convert data to appropriate sEMG images. (For example: [batch_size, 1, 8(sensors/channels), 52(window size)])
                sampled_inputs = sampled_inputs.reshape(-1, 8, window, 1)
                
                ## Optional cast data to float 32
                sampled_inputs = sampled_inputs.astype(np.float32)
                
                ## Train model
                self.model.fit(
                    sampled_inputs,
                    sampled_outputs,
                    batch_size=realtime_batch_size,
                    epochs=realtime_epochs
                )
                if finetuned_path != None:
                    self.model.save_weights(finetuned_path)
                
                ## Predict gestures until network is disconnected
                
                while True:
                    if not self.connected:
                        break
                    await self.client.start_notify(self.EMG0, self.prediction_handler)
                    await self.client.start_notify(self.EMG1, self.prediction_handler)
                    await self.client.start_notify(self.EMG2, self.prediction_handler)
                    await self.client.start_notify(self.EMG3, self.prediction_handler)
                    
                    
            else:
                print(f"Failed to connect to {self.connected_device.name}")
        except Exception as e:
            print(e)

    # ##
    #     Selects and connects to a BLE device
    
    # ##
    async def select_device(self):
        print("Bluetooh LE hardware warming up...")
        await asyncio.sleep(2.0, loop=loop)
        ##Searches for BLE devices
        devices = await discover()
       
        response = None
        for i, device in enumerate(devices):
            if device.name == "Cyclops":
                response = i
        
        if response == None:
            print("Could not find myo armband. Please Try Again.")
            self.cleanup()
                
        print(f"Connecting to {devices[response].name}")
        self.connected_device = devices[response]
        self.client = BleakClient(devices[response].address, loop=self.loop)
    

    ## Handler for collecting 2 Sequential Sequence from Myo Armaband (MyoCharacteristics0)
    ## Each Sequential Sequence contains 1 sample from each 8 sensors/channels
    def training_handler(self, sender: str, data: Any):
        sequence_1, sequence_2 = getFeatures(data, twos_complement=True)
        for channel_idx in range(8):
            sensors[channel_idx].append(sequence_1[channel_idx])
            sensors[channel_idx].append(sequence_2[channel_idx])
  
  
    ## Handler for collecting 2 Sequential Sequence from Myo Armaband
    ## Each Sequential Sequence contains 1 sample from each 8 sensors/channels
    async def prediction_handler(self, sender: str, data: Any):
        sequence_1, sequence_2 = getFeatures(data, twos_complement=True)
        for channel_idx in range(8):
            self.current_sample[channel_idx].append(sequence_1[channel_idx])
            self.current_sample[channel_idx].append(sequence_2[channel_idx])
        
        if len(self.current_sample[0]) >= window:
            ## Truncate self.current_samples to window size
            sEMG = np.array([samples[-window:] for samples in self.current_sample])
            
            ## Apply Standarization to sEMG data
            for channel_idx in range(len(sEMG)):
                mean = params[str(channel_idx)][0]
                std = params[str(channel_idx)][1]
                sEMG[channel_idx] = (sEMG[channel_idx] - mean) / std
            
            ## Optional cast input to float 32 (demand of microcontroller)
            sEMG = sEMG.astype(np.float32)
            
            ## Get prediction results
            pred = realtime_pred(
                self.model,
                sEMG,
                num_channels=num_sensors,
                window_length=window
            )
            
            
            ## Update prediction results
            print(GESTURES[pred])
            
            ## Remove first 8 instance from self.current_samples to collect new data. (overlaps)
            self.current_sample = [samples[-(window-step_size):] for samples in self.current_sample]
            

def getFeatures(data, twos_complement=True):
    sequence_1 = []
    sequence_2 = []
    for i in range(8):
        if twos_complement==True and data[i] > 127:
            sequence_1.append(data[i]-256)
        else:
            sequence_1.append(data[i])
            
    for i in range(8,16):
        if twos_complement==True and data[i] > 127:
            sequence_2.append(data[i]-256)
        else:
            sequence_2.append(data[i])

    return sequence_1, sequence_2

##########################
## App Main
##########################
if __name__ == "__main__":

    ## Create the event loop.
    loop = asyncio.get_event_loop()
    connection = Connection(loop, EMG0, EMG1, EMG2, EMG3, CONTROL) ## EMG3
    try:
        asyncio.ensure_future(connection.manager())
        loop.run_forever()
    except KeyboardInterrupt:
        print("User stopped program.")
    finally:
        print("Disconnecting...")
        loop.run_until_complete(connection.cleanup())
