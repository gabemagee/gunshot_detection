#!/usr/bin/env python
# coding: utf-8

# ## Package Imports

# In[ ]:


import pyaudio
import librosa
import logging
import time
import wave
import scipy.signal
import IPython.display as ipd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from threading import Thread
from array import array
from scipy.io import wavfile
from queue import Queue
from sklearn.preprocessing import LabelBinarizer
# from tensorflow.keras import Input, layers, optimizers, backend as K
# from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
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


audio_format = pyaudio.paFloat32
audio_rate = 44100
audio_channels = 1
audio_device_index = 6
audio_frames_per_buffer = 4410
audio_sample_duration = 2
audio_analysis_queue = Queue()
sound_data = np.zeros(0, dtype = "float32")
audio_volume_threshold = 0.5
sms_alert_queue = Queue()
inference_model_confidence_threshold = 0.95
max_audio_frame_int_value = 2 ** 15 - 1
sound_normalization_threshold = 10 ** (-1.0 / 20)
designated_alert_recipients = ["8163449956", "9176202840", "7857642331"]


# ## Loading in Augmented Labels

# In[ ]:


labels = np.load("/home/pi/Datasets/gunshot_augmented_sound_labels.npy")


# ## Binarizing Labels

# In[ ]:


labels = np.array([("gun_shot" if label == 1 else "other") for label in labels])
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)
labels = np.hstack((labels, 1 - labels))


# ## Sound Post-Processing Functions

# In[ ]:


def normalize(sound_data):
    normalization_factor = float(sound_normalization_threshold * max_audio_frame_int_value) / max(abs(i) for i in sound_data)
    
    # Averages the volume out
    r = array('f')
    for datum in sound_data:
        r.append(int(datum * normalization_factor))
    return np.array(r, dtype = np.float32)


# ### Librosa Wrapper Function Definitions

# In[ ]:


def _stft(y, n_fft, hop_length, win_length):
    return librosa.stft(y = y, n_fft = n_fft, hop_length = hop_length, win_length = win_length)


def _istft(y, hop_length, win_length):
    return librosa.istft(y, hop_length, win_length)


def _amp_to_db(x):
    return librosa.core.logamplitude(x, ref_power = 1.0, amin = 1e-20, top_db = 80.0)  # Librosa 0.4.2 functionality
#     return librosa.core.amplitude_to_db(x, ref = 1.0, amin = 1e-20, top_db = 80.0)  # Librosa 0.6.3 functionality


def _db_to_amp(x):
    return librosa.core.perceptual_weighting(x, frequencies = 1.0)  # Librosa 0.4.2 functionality
#     return librosa.core.db_to_amplitude(x, ref = 1.0)  # Librosa 0.6.3 functionality


# ### Custom Noise Reduction Function Definition

# In[ ]:


def remove_noise(audio_clip,
                noise_clip,
                n_grad_freq = 2,
                n_grad_time = 4,
                n_fft = 2048,
                win_length = 2048,
                hop_length = 512,
                n_std_thresh = 1.5,
                prop_decrease = 1.0,
                verbose = False,
                visual = False):
    
    """ Removes noise from audio based upon a clip containing only noise

    Args:
        audio_clip (array): The first parameter.
        noise_clip (array): The second parameter.
        n_grad_freq (int): how many frequency channels to smooth over with the mask.
        n_grad_time (int): how many time channels to smooth over with the mask.
        n_fft (int): number audio of frames between STFT columns.
        win_length (int): Each frame of audio is windowed by `window()`. The window will be of length `win_length` and then padded with zeros to match `n_fft`..
        hop_length (int):number audio of frames between STFT columns.
        n_std_thresh (int): how many standard deviations louder than the mean dB of the noise (at each frequency level) to be considered signal
        prop_decrease (float): To what extent should you decrease noise (1 = all, 0 = none)
        visual (bool): Whether to plot the steps of the algorithm

    Returns:
        array: The recovered signal with noise subtracted

    """
    
    # Debugging
    if verbose:
        start = time.time()
        
    # Takes a STFT over the noise sample
    noise_stft = _stft(noise_clip, n_fft, hop_length, win_length)
    noise_stft_db = _amp_to_db(np.abs(noise_stft))  # Converts the sample units to dB
    
    # Calculates statistics over the noise sample
    mean_freq_noise = np.mean(noise_stft_db, axis = 1)
    std_freq_noise = np.std(noise_stft_db, axis = 1)
    noise_thresh = mean_freq_noise + std_freq_noise * n_std_thresh
    
    # Debugging
    if verbose:
        print("STFT on noise:", td(seconds = time.time() - start))
        start = time.time()
        
    # Takes a STFT over the signal sample
    sig_stft = _stft(audio_clip, n_fft, hop_length, win_length)
    sig_stft_db = _amp_to_db(np.abs(sig_stft))
    
    # Debugging
    if verbose:
        print("STFT on signal:", td(seconds = time.time() - start))
        start = time.time()
        
    # Calculates value to which to mask dB
    mask_gain_dB = np.min(_amp_to_db(np.abs(sig_stft)))
    
    # Debugging
    if verbose:
        print("Noise Threshold & Mask Gain in dB: ", noise_thresh, mask_gain_dB)
    
    # Creates a smoothing filter for the mask in time and frequency
    smoothing_filter = np.outer(
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_freq + 1, endpoint = False),
                np.linspace(1, 0, n_grad_freq + 2),
            ]
        )[1:-1],
        np.concatenate(
            [
                np.linspace(0, 1, n_grad_time + 1, endpoint = False),
                np.linspace(1, 0, n_grad_time + 2),
            ]
        )[1:-1]
    )
    
    smoothing_filter = smoothing_filter / np.sum(smoothing_filter)
    
    # Calculates the threshold for each frequency/time bin
    db_thresh = np.repeat(np.reshape(noise_thresh, [1, len(mean_freq_noise)]),
                          np.shape(sig_stft_db)[1],
                          axis = 0).T
    
    # Masks segment if the signal is above the threshold
    sig_mask = sig_stft_db < db_thresh
    
    # Debugging
    if verbose:
        print("Masking:", td(seconds=time.time() - start))
        start = time.time()
        
    # Convolves the mask with a smoothing filter
    sig_mask = scipy.signal.fftconvolve(sig_mask, smoothing_filter, mode="same")
    sig_mask = sig_mask * prop_decrease
    
    # Debugging
    if verbose:
        print("Mask convolution:", td(seconds=time.time() - start))
        start = time.time()
        
    # Masks the signal
    sig_stft_db_masked = (sig_stft_db * (1 - sig_mask)
                          + np.ones(np.shape(mask_gain_dB))
                          * mask_gain_dB * sig_mask)  # Masks real
    
    sig_imag_masked = np.imag(sig_stft) * (1 - sig_mask)
    sig_stft_amp = (_db_to_amp(sig_stft_db_masked) * np.sign(sig_stft)) + (1j * sig_imag_masked)
    
    # Debugging
    if verbose:
        print("Mask application:", td(seconds=time.time() - start))
        start = time.time()
        
    # Recovers the signal
    recovered_signal = _istft(sig_stft_amp, hop_length, win_length)
    recovered_spec = _amp_to_db(
        np.abs(_stft(recovered_signal, n_fft, hop_length, win_length))
    )
    
    # Debugging
    if verbose:
        print("Signal recovery:", td(seconds=time.time() - start))
        
    # Returns noise-reduced audio sample
    return recovered_signal


# ### WAV File Composition Function

# In[ ]:


# Saves a two-second gunshot sample as a WAV file
def create_gunshot_wav_file(microphone_data, index, timestamp, number_of_audio_channels = audio_channels, sample_width = 2, frame_rate = 44100):
    microphone_data = microphone_data.reshape(44100)
    librosa.output.write_wav("./recordings/Gunshot Sound Sample #"
                            + str(index) + " ("
                            + str(timestamp) + ").wav", microphone_data, 22050)
    
#     wav_file = wave.open("./recordings/Gunshot Sound Sample #"
#                             + str(index) + " ("
#                             + str(timestamp) + ").wav", "wb")
#     wav_file.setnchannels(number_of_audio_channels)
#     wav_file.setsampwidth(sample_width)
#     wav_file.setframerate(frame_rate)
#     wav_file.writeframes(np.int32(microphone_data / np.max(np.abs(microphone_data)) * 32767))
#     wav_file.close()


# ## Loading in Noise Sample

# In[ ]:


noise_sample_wav = "../noise_reduction/Noise Sample - Sizheng Microphone.wav"
noise_sample_rate, noise_sample = wavfile.read(noise_sample_wav)
noise_clip = noise_sample  # In this case, the whole sample is a clip of noise


# ## Model Construction Functions

# #### ROC (AUC) metric - Uses the import "from tensorflow.keras import backend as K"

# In[ ]:


# def auc(y_true, y_pred):
#     auc = tf.metrics.auc(y_true, y_pred)[1]
#     K.get_session().run(tf.local_variables_initializer())
#     return auc


# #### 1D Time-Series Model

# In[ ]:


# def load_model_one(weights_file):
#     # Initializing 1D Time-Series Model Parameters
#     drop_out_rate = 0.1
#     learning_rate = 0.001
#     number_of_epochs = 100
#     number_of_classes = 2
#     batch_size = 32
#     optimizer = optimizers.Adam(learning_rate, learning_rate / 100)
#     input_shape = (44100, 1)
#     input_tensor = Input(shape = input_shape)
#     metrics = [auc, "accuracy"]
    
#     # Reconstructing 1D Time-Series Model
#     x = layers.Conv1D(16, 9, activation = "relu", padding = "same")(input_tensor)
#     x = layers.Conv1D(16, 9, activation = "relu", padding = "same")(x)
#     x = layers.MaxPool1D(16)(x)
#     x = layers.Dropout(rate = drop_out_rate)(x)

#     x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
#     x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
#     x = layers.MaxPool1D(4)(x)
#     x = layers.Dropout(rate = drop_out_rate)(x)

#     x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
#     x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
#     x = layers.MaxPool1D(4)(x)
#     x = layers.Dropout(rate = drop_out_rate)(x)

#     x = layers.Conv1D(256, 3, activation = "relu", padding = "same")(x)
#     x = layers.Conv1D(256, 3, activation = "relu", padding = "same")(x)
#     x = layers.GlobalMaxPool1D()(x)
#     x = layers.Dropout(rate = (drop_out_rate * 2))(x) # Increasing drop-out rate here to prevent overfitting

#     x = layers.Dense(64, activation = "relu")(x)
#     x = layers.Dense(1028, activation = "relu")(x)
    
#     # Compiling 1D Time-Series Model
#     output_tensor = layers.Dense(number_of_classes, activation = "softmax")(x)
#     model = tf.keras.Model(input_tensor, output_tensor)
#     model.compile(optimizer = optimizer, loss = keras.losses.binary_crossentropy, metrics = metrics)
    
#     # Loading 1D Time-Series Model Weights
#     model.load_weights(weights_file)
    
#     return model


# #### 2D Spectrogram Model

# In[ ]:


# def load_model_two(weights_file):
#     # 2D Spectrogram Model Parameters
#     input_shape = (128, 87, 1)
#     input_tensor = Input(shape = input_shape)
#     learning_rate = 0.001
#     optimizer = optimizers.Adam(learning_rate, learning_rate / 100)
#     filter_size = (3,3)
#     maxpool_size = (3,3)
#     activation = "relu"
#     drop_out_rate = 0.1
#     number_of_classes = 2
#     metrics = [auc, "accuracy"]
    
#     # Reconstructing 2D Spectrogram Model
#     x = layers.Conv2D(16, filter_size, activation = activation, padding = "same")(input_tensor)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPool2D(maxpool_size)(x)
#     x = layers.Dropout(rate = drop_out_rate)(x)

#     x = layers.Conv2D(32, filter_size, activation = activation, padding = "same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPool2D(maxpool_size)(x)
#     x = layers.Dropout(rate = drop_out_rate)(x)

#     x = layers.Conv2D(64, filter_size, activation = activation, padding = "same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.MaxPool2D(maxpool_size)(x)
#     x = layers.Dropout(rate = drop_out_rate)(x)

#     x = layers.Conv2D(256, filter_size, activation = activation, padding = "same")(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.GlobalMaxPool2D()(x)
#     x = layers.Dropout(rate = (drop_out_rate * 2))(x) # Increasing drop-out rate here to prevent overfitting

#     x = layers.Dense(64, activation = activation)(x)
#     x = layers.Dense(1028, activation = activation)(x)
    
#     # Compiling 2D Spectrogram Model
#     output_tensor = layers.Dense(number_of_classes, activation = "softmax")(x)
#     spec_model = tf.keras.Model(input_tensor, output_tensor)
#     spec_model.compile(optimizer = optimizer, loss = keras.losses.binary_crossentropy, metrics = metrics)

#     # Loading 2D Spectrogram Model Weights
#     spec_model.load_weights(weights_file)
    
#     return spec_model


# ## ---

# # Multithreaded Inference: A callback thread adds two second samples of microphone data to the audio analysis queue; The main thread, an audio analysis thread, detects the presence of gunshot sounds in samples retrieved from the audio analysis queue; And an SMS alert thread dispatches groups of messages to designated recipients.

# ## ---

# ## Defining Threads

# ### SMS Alert Thread

# In[ ]:


def send_sms_alert():
    # Continuously dispatches SMS alerts to a list of designated recipients
    while True:
        sms_alert_status = sms_alert_queue.get()
        if sms_alert_status == "Gunshot Detected":
            logger.debug("--- ALERT: A Gunshot Has Been Detected ---")
    
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
    
    # The SMS alert thread will run indefinitely
    while True:
        sms_alert_status = sms_alert_queue.get()
        if sms_alert_status == "Gunshot Detected":
            try:
                # At this point in execution, an attempt to send an SMS alert to local authorities will be made
                modem.waitForNetworkCoverage(timeout = 86400)
                message = "--- ALERT: A Gunshot Has Been Detected ---"
                for number in designated_alert_recipients:
                    modem.sendSms(number, message)
                logger.debug(" *** Sent out an SMS alert to all designated recipients *** ")
            except:
                logger.debug("ERROR: Unable to successfully send an SMS alert to the designated recipients.")
                pass
            finally:
                logger.debug(" ** Finished evaluating an audio sample with the model ** ")
    
    """

# Starts the SMS alert thread
sms_alert_thread = Thread(target = send_sms_alert)
sms_alert_thread.start()


# ### Callback Thread

# In[ ]:


def callback(in_data, frame_count, time_info, status):
    global sound_data
    sound_buffer = np.frombuffer(in_data, dtype = "float32")
    sound_data = np.append(sound_data, sound_buffer)
    if len(sound_data) >= 88200:
        audio_analysis_queue.put(sound_data)
        current_time = time.ctime(time.time())
        audio_analysis_queue.put(current_time)
    return (sound_buffer, pyaudio.paContinue)

pa = pyaudio.PyAudio()

stream = pa.open(format = audio_format,
                 rate = audio_rate,
                 channels = audio_channels,
                 input_device_index = audio_device_index,
                 frames_per_buffer = audio_frames_per_buffer,
                 input = True,
                 stream_callback = callback)

# Starts the callback thread
stream.start_stream()
logger.debug("--- Listening to Audio Stream ---")


# ## Pre-Inference Debugging of Captured Audio Samples (Optional)

# In[ ]:


# # Loads TFLite model and allocate tensors
# interpreter = tf.lite.Interpreter(model_path = "./models/gunshot_sound_model.tflite")
# interpreter.allocate_tensors()

# # Gets input and output tensors as well as the input shape
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# input_shape = input_details[0]['shape']

# # Gets a sample and its timestamp from the audio analysis queue
# microphone_data = np.array(audio_analysis_queue.get(), dtype = "float32")

# # Post-processes the microphone data
# modified_microphone_data = librosa.resample(y = microphone_data, orig_sr = audio_rate, target_sr = 22050)
# modified_microphone_data = normalize(modified_microphone_data)
# # modified_microphone_data = remove_noise(audio_clip = modified_microphone_data, noise_clip = noise_clip)  # As a substitute for normalization
# # number_of_missing_hertz = 44100 - len(modified_microphone_data)
# # modified_microphone_data = np.array(modified_microphone_data.tolist() + [0 for i in range(number_of_missing_hertz)], dtype = "float32")
# modified_microphone_data = modified_microphone_data[:44100]

# ipd.Audio(modified_microphone_data, rate = 22050)


# ### Main (Audio Analysis) Thread

# In[ ]:


# Loads TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path = "./models/gunshot_sound_model.tflite")
interpreter.allocate_tensors()

# Gets input and output tensors as well as the input shape
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Loading 1D Time-Series Model
# model = load_model_one("./models/gunshot_sound_model.h5")
    
# Loading 2D Spectrogram Model
#   model = load_model_two("./models/gunshot_sound_model_spectrograph_model.h5")
    
# An iterator variable for counting the number of gunshot sounds detected
gunshot_sound_counter = 1

# The main (audio analysis) thread will run indefinitely
while True:
    # Gets a sample and its timestamp from the audio analysis queue
    microphone_data = np.array(audio_analysis_queue.get(), dtype = "float32")
    time_of_sample_occurrence = audio_analysis_queue.get()
    
    # Cleans up the global NumPy audio data source
    sound_data = np.zeros(0, dtype = "float32")
        
    # Finds the current sample's maximum frequency value
    maximum_frequency_value = np.max(microphone_data)
        
    # Determines whether a given sample potentially contains a gunshot
    if maximum_frequency_value >= audio_volume_threshold:
        
        # Displays the current sample's maximum frequency value
        logger.debug("The maximum frequency value of a given sample before processing: " + str(maximum_frequency_value))
        
        # Post-processes the microphone data
        modified_microphone_data = librosa.resample(y = microphone_data, orig_sr = audio_rate, target_sr = 22050)
        modified_microphone_data = normalize(modified_microphone_data)
#         modified_microphone_data = remove_noise(audio_clip = modified_microphone_data, noise_clip = noise_clip)  # As a substitute for normalization
#         number_of_missing_hertz = 44100 - len(modified_microphone_data)
#         modified_microphone_data = np.array(modified_microphone_data.tolist() + [0 for i in range(number_of_missing_hertz)], dtype = "float32")
        modified_microphone_data = modified_microphone_data[:44100]
        modified_microphone_data = modified_microphone_data.reshape(input_shape)

        # Passes a given audio sample into the model for prediction
#         probabilities = model.predict(modified_microphone_data)
        interpreter.set_tensor(input_details[0]["index"], modified_microphone_data)
        interpreter.invoke()
        probabilities = interpreter.get_tensor(output_details[0]["index"])
        logger.debug("The model-predicted probability values: " + str(probabilities[0]))
        logger.debug("Model-predicted sample class: " + label_binarizer.inverse_transform(probabilities[:, 0])[0])

        # Determines if a gunshot sound was detected by the model
        if (probabilities[0][1] >= inference_model_confidence_threshold):
            # Sends out an SMS alert
            sms_alert_queue.put("Gunshot Detected")
            
            # Makes a WAV file of the gunshot sample
            create_gunshot_wav_file(modified_microphone_data, gunshot_sound_counter, time_of_sample_occurrence)

            # Increments the counter for gunshot sound file names
            gunshot_sound_counter += 1


# ## Testing A Model with Sample Audio (Optional)

# In[ ]:


# # Loads TFLite model and allocate tensors
# interpreter = tf.lite.Interpreter(model_path = "./models/gunshot_sound_model.tflite")
# interpreter.allocate_tensors()

# # Gets input and output tensors as well as the input shape
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
# input_shape = input_details[0]['shape']

# # Loads in test sample WAV file
# gunshot_training_sample, sr = librosa.load("./recordings/260600_8.wav")
# # training_sample = normalize(training_sample)
# number_of_missing_hertz = 44100 - len(gunshot_training_sample)
# gunshot_training_sample = np.array(gunshot_training_sample.tolist() + [0 for i in range(number_of_missing_hertz)], dtype = "float32")
# gunshot_training_sample = gunshot_training_sample.reshape(input_shape)


# In[ ]:


# # Performs inference with the TensorFlow Lite model
# interpreter.set_tensor(input_details[0]["index"], gunshot_training_sample)
# # interpreter.set_tensor(input_details[0]["index"], np.array(np.random.random_sample(input_shape), dtype = "float32"))
# interpreter.invoke()
# probabilities = interpreter.get_tensor(output_details[0]["index"])
# logger.debug("The model-predicted probability values: " + str(probabilities[0]))
# logger.debug("Model-predicted sample class: " + label_binarizer.inverse_transform(probabilities[:, 0])[0])


# In[ ]:




