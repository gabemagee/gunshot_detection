import tensorflow as tf
import os

saved_model_dir = os.getcwd() + "/raspberry_pi/models/new_spectrogram_model.h5"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
