
# coding: utf-8

# ## Package Imports

# In[ ]:

import pyaudio
import librosa
import logging
import sys
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from multiprocessing import Process
from tensorflow.keras import Input, layers, optimizers, backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from gsmmodem.modem import GsmModem
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# ## Variable Initializations

# In[ ]:

audio_format = pyaudio.paFloat32
audio_rate = 44100
audio_channels = 1
audio_device_index = 1
audio_frames_per_buffer = 4410
audio_sample_duration = 2
input_shape = (audio_rate, 1)
modem_port = '/dev/ttyUSB0'
modem_baudrate = 115200
modem_sim_pin = None  # SIM card PIN (if any)
phone_numbers_to_message = ["8163449956", "9176202840", "7857642331"]

# ## Establishing a Connection to the SMS Modem

# In[ ]:

logging.info("Initializing connection to modem...")
modem = GsmModem(modem_port, modem_baudrate)
modem.smsTextMode = False
modem.connect(modem_sim_pin)


# ## ROC (AUC) metric - Uses the import "from tensorflow.keras import backend as K"

# In[ ]:

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


# ## Defining Spectrogram Model

# In[ ]:

#input_shape = (128, 87, 1)
learning_rate = 0.001
optimizer = optimizers.Adam(learning_rate, learning_rate / 100)
#input_tensor = Input(shape=input_shape)
#filter_size = (3,3)
#maxpool_size = (3,3)
#activation = "relu"
drop_out_rate = 0.1
#number_of_classes = 2
metrics = [auc, "accuracy"]

# Model Architecture
#x = layers.Conv2D(16, filter_size, activation=activation, padding="same")(input_tensor)
#x = layers.BatchNormalization()(x)
#x = layers.MaxPool2D(maxpool_size)(x)
#x = layers.Dropout(rate=drop_out_rate)(x)

#x = layers.Conv2D(32, filter_size, activation=activation, padding="same")(x)
#x = layers.BatchNormalization()(x)
#x = layers.MaxPool2D(maxpool_size)(x)
#x = layers.Dropout(rate=drop_out_rate)(x)

#x = layers.Conv2D(64, filter_size, activation=activation, padding="same")(x)
#x = layers.BatchNormalization()(x)
#x = layers.MaxPool2D(maxpool_size)(x)
#x = layers.Dropout(rate=drop_out_rate)(x)

#x = layers.Conv2D(256, filter_size, activation=activation, padding="same")(x)
#x = layers.BatchNormalization()(x)
#x = layers.GlobalMaxPool2D()(x)
#x = layers.Dropout(rate=(drop_out_rate * 2))(x) # Increasing drop-out rate here to prevent overfitting

#x = layers.Dense(64, activation=activation)(x)
#x = layers.Dense(1028, activation=activation)(x)
#output_tensor = layers.Dense(number_of_classes, activation="softmax")(x)

#spec_model = tf.keras.Model(input_tensor, output_tensor)
#spec_model.compile(optimizer=optimizer, loss=keras.losses.binary_crossentropy, metrics=metrics)


# ## Loading Spectrogram Model Weights

# In[ ]:

#spec_model.load_weights("./models/gunshot_sound_model_spectrograph_model.h5")


# ## Original Model Parameters

# In[ ]:

number_of_epochs = 100
number_of_classes = 2
batch_size = 32
sample_rate_per_two_seconds = 44100
input_shape = (sample_rate_per_two_seconds, 1)
input_tensor = Input(shape=input_shape)
# ## Loading Original Model

# In[ ]:

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


# ## Loading Original Model Weights

# In[ ]:

model.load_weights("./models/gunshot_sound_model.h5")


# ## Defining Process Functions

# In[ ]:

def analyze_microphone_data(microphone_data, audio_rate, model, modem, phone_numbers_to_message):
    # Performs post-processing on live audio samples
    reformed_microphone_data = librosa.resample(y=microphone_data, orig_sr=audio_rate, target_sr=22050)
    reformed_microphone_data = librosa.util.normalize(reformed_microphone_data)
    reformed_microphone_data = reformed_microphone_data[:audio_rate]
    reformed_microphone_data = reformed_microphone_data.reshape(-1, audio_rate, 1)
        
    # Passes a given audio sample into the model for prediction
    probabilities = model.predict(reformed_microphone_data)
    logging.info("Probabilities derived by the model:%s", probabilities)
    if (probabilities[0][1] >= 0.9):
        sms_alert_process = Process(target = send_sms_alert, args = (probabilities))
        sms_alert_process.start()


def send_sms_alert(probabilities):
        # If the model detects a gunshot, an SMS alert will be sent to local authorities
        try:
            modem.waitForNetworkCoverage(timeout=86400)
            message = " (Testing) ALERT: A Gunshot Has Been Detected (Testing)"
            for number in phone_numbers_to_message:
                modem.sendSms(number, message)
            logging.info(" *** Sent out an SMS alert to all designated recipients *** ")
        except:
            logging.info("ERROR: Unable to successfully send an SMS alert to the designated recipients.")
            pass
        finally:
            logging.info(" * Finished evaluating an audio sample with the model * ")


# ## Capturing Microphone Audio

# In[ ]:

pa = pyaudio.PyAudio()
    
stream = pa.open(format = audio_format,
                 rate = audio_rate,
                 channels = audio_channels,
                 input_device_index = audio_device_index,
                 frames_per_buffer = audio_frames_per_buffer,
                 input = True)

logging.info("--- Recording Audio ---")
while(True):
    np_array_data = []
    
    # Loops through the stream and appends audio chunks to the frame array
    for i in range(0, int(audio_rate / audio_frames_per_buffer * audio_sample_duration)):
        data = stream.read(audio_frames_per_buffer, exception_on_overflow = False)
        np_array_data.append(np.frombuffer(data, dtype=np.float32))
    microphone_data = np.concatenate(np_array_data)
    logging.info("Cumulative length of a given two-second audio sample:%s", len(microphone_data))
    logging.info("The maximum frequency value for a given two-second audio sample:%s", max(microphone_data))
    
    # If a sample meets a certain threshold, a new concurrent analysis process is created
    if max(microphone_data) >= 0.001:
        analysis_process = Process(target = analyze_microphone_data, args = (microphone_data, audio_rate, model, modem, phone_numbers_to_message))
        analysis_process.start()
        
    # Closes all finished processes    
    multiprocessing.active_children()
