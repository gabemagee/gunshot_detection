#!/usr/bin/env python
# coding: utf-8

# ## Package Imports

# In[ ]:


import pyaudio
import librosa
import logging
import time
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Input, layers, optimizers, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
#from gsmmodem.modem import GsmModem


# ## Configuring the Logger

# In[ ]:


logger = logging.getLogger('debugger')
logger.setLevel(logging.DEBUG)
ch = logging.FileHandler('output.log')
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


# ## Variable Initializations

# In[ ]:


audio_format = pyaudio.paInt16
audio_rate = 44100
audio_channels = 1
audio_device_index = 6  # For personal laptop
audio_frames_per_buffer = 22050  # New experimental value
audio_sample_duration = 2
phone_numbers_to_message = ["8163449956", "9176202840", "7857642331"]


# ## Defining Process Functions

# In[ ]:


# Multiprocessing Inference: Currently, there is one audio analysis process running for the duration of the program
# The main process adds the microphone data to the audio analysis queue which the audio analysis process then retrieves and analyzes
# If the audio analysis process detects the sound of a gunshot, the audio analysis queue will add a "1" to the queue, signifying the detection of a gunshot

def analyze_microphone_data(audio_rate):
    
    # ROC (AUC) metric - Uses the import "from tensorflow.keras import backend as K"
    def auc(y_true, y_pred):
        auc = tf.metrics.auc(y_true, y_pred)[1]
        K.get_session().run(tf.local_variables_initializer())
        return auc
    
    # 1D Time-Series Model Parameters
    drop_out_rate = 0.1
    learning_rate = 0.001
    number_of_epochs = 100
    number_of_classes = 2
    batch_size = 32
    optimizer = optimizers.Adam(learning_rate, learning_rate / 100)
    input_shape = (audio_rate, 1)
    input_tensor = Input(shape=input_shape)
    metrics = [auc, "accuracy"]

    # Loading 1D Time-Series Model
    x = layers.Conv1D(16, 9, activation="relu", padding="same")(input_tensor)
    x = layers.Conv1D(16, 9, activation="relu", padding="same")(x)
    x = layers.MaxPool1D(16)(x)
    x = layers.Dropout(rate=drop_out_rate)(x)

    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool1D(4)(x)
    x = layers.Dropout(rate=drop_out_rate)(x)

    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPool1D(4)(x)
    x = layers.Dropout(rate=drop_out_rate)(x)

    x = layers.Conv1D(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv1D(256, 3, activation="relu", padding="same")(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dropout(rate=(drop_out_rate * 2))(x) # Increasing drop-out rate here to prevent overfitting

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(1028, activation="relu")(x)
    output_tensor = layers.Dense(number_of_classes, activation="softmax")(x)

    model = tf.keras.Model(input_tensor, output_tensor)
    model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=metrics)
    
    # Loading 1D Time-Series Model Weights
    model.load_weights("./models/gunshot_sound_model.h5")
    
    # 2D Spectrogram Model Parameters
#     input_shape = (128, 87, 1)
#     input_tensor = Input(shape=input_shape)
#     learning_rate = 0.001
#     optimizer = optimizers.Adam(learning_rate, learning_rate / 100)
#     filter_size = (3,3)
#     maxpool_size = (3,3)
#     activation = "relu"
#     drop_out_rate = 0.1
#     number_of_classes = 2
#     metrics = [auc, "accuracy"]

    # Loading 2D Spectrogram Model
#     x = layers.Conv2D(16, filter_size, activation=activation, padding="same")(input_tensor)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPool2D(maxpool_size)(x)
#     x = layers.Dropout(rate=drop_out_rate)(x)

#     x = layers.Conv2D(32, filter_size, activation=activation, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPool2D(maxpool_size)(x)
#     x = layers.Dropout(rate=drop_out_rate)(x)

#     x = layers.Conv2D(64, filter_size, activation=activation, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPool2D(maxpool_size)(x)
#     x = layers.Dropout(rate=drop_out_rate)(x)

#     x = layers.Conv2D(256, filter_size, activation=activation, padding="same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.GlobalMaxPool2D()(x)
#     x = layers.Dropout(rate=(drop_out_rate * 2))(x) # Increasing drop-out rate here to prevent overfitting

#     x = layers.Dense(64, activation=activation)(x)
#     x = layers.Dense(1028, activation=activation)(x)
#     output_tensor = layers.Dense(number_of_classes, activation="softmax")(x)

#     spec_model = tf.keras.Model(input_tensor, output_tensor)
#     spec_model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=metrics)

    # Loading 2D Spectrogram Model Weights
#     spec_model.load_weights("./models/gunshot_sound_model_spectrograph_model.h5")

    # An iterator variable for counting the number of gunshot sounds detected
    gunshot_sound_counter = 1
    
    # The audio analysis process will run indefinitely
    while True:
        
        # Waits to continue until something is in the queue
        microphone_data = audio_analysis_queue.get()
        
        # Performs post-processing on live audio samples
        reformed_microphone_data = librosa.resample(y=microphone_data, orig_sr=audio_rate, target_sr=22050)
        reformed_microphone_data = librosa.util.normalize(reformed_microphone_data)
        reformed_microphone_data = reformed_microphone_data[:audio_rate]
        reformed_microphone_data = reformed_microphone_data.reshape(-1, audio_rate, 1)

        # Passes a given audio sample into the model for prediction
        probabilities = model.predict(reformed_microphone_data)
        logger_message = "Probabilities derived by the model: " + str(probabilities)
        logger.debug(logger_message)
        if (probabilities[0][1] >= 0.9):
            sms_alert_queue.put(1)
            librosa.output.write_wav("Gunshot Sound Sample #" + str(gunshot_sound_counter) + ".wav", microphone_data, 22050)
            gunshot_sound_counter += 1


def send_sms_alert(phone_numbers_to_message):
    
    return 0
    
    """
    
    # Configuring the Modem Connection
    modem_port = '/dev/ttyUSB0'
    modem_baudrate = 115200
    modem_sim_pin = None  # SIM card PIN (if any)
    
    # Establishing a Connection to the SMS Modem
    logger.debug("Initializing connection to modem...")
    modem = GsmModem(modem_port, modem_baudrate)
    modem.smsTextMode = False
    modem.connect(modem_sim_pin)
    
    # The SMS alert process will run indefinitely
    while True:
        sms_alert_status = sms_alert_queue.get()
        if sms_alert_status == 1:
            try:
                # At this point in execution, an attempt to send an SMS alert to local authorities will be made
                modem.waitForNetworkCoverage(timeout=86400)
                message = " (Testing) ALERT: A Gunshot Has Been Detected (Testing)"
                for number in phone_numbers_to_message:
                    modem.sendSms(number, message)
                logger.debug(" *** Sent out an SMS alert to all designated recipients *** ")
            except:
                logger.debug("ERROR: Unable to successfully send an SMS alert to the designated recipients.")
                pass
            finally:
                logger.debug(" ** Finished evaluating an audio sample with the model ** ")
    
    """
    


# ## Opening the Microphone Audio Stream

# In[ ]:


pa = pyaudio.PyAudio()
    
stream = pa.open(format = audio_format,
                 rate = audio_rate,
                 channels = audio_channels,
                 input_device_index = audio_device_index,
                 frames_per_buffer = audio_frames_per_buffer,
                 input = True)


# ## Capturing Microphone Audio

# In[ ]:


logger.debug("--- Listening to Audio Stream ---")

audio_analysis_process = multiprocessing.Process(target = analyze_microphone_data, args = (audio_rate,))
sms_alert_process = multiprocessing.Process(target = send_sms_alert, args = (phone_numbers_to_message,))
audio_analysis_queue = multiprocessing.Queue()
sms_alert_queue = multiprocessing.Queue()
audio_analysis_process.start()
sms_alert_process.start()

while True:
    sound_data = []
    
    # Loops through the stream and appends audio chunks to the frame array
    for i in range(0, int(audio_rate / audio_frames_per_buffer * audio_sample_duration)):
        sound_buffer = stream.read(audio_frames_per_buffer, exception_on_overflow = False)
        sound_data.append(np.frombuffer(sound_buffer, dtype=np.int16))
    microphone_data = np.concatenate(sound_data)
    logger_message = "The maximum frequency value for a given two-second audio sample: " + str(max(microphone_data))
    logger.debug(logger_message)
    
    # If a sample meets a certain threshold, a new batch of microphone data is placed on the queue
    if max(microphone_data) >= 500:
        audio_analysis_queue.put(microphone_data)
        
    # Closes all finished processes   
    multiprocessing.active_children()

