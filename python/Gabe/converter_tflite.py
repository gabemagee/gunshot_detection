import tensorflow as tf
import os

saved_model_dir = "/home/gamagee/workspace/gunshot_detection/raspberry_pi/models/new_spectrogram_model.h5"
converter = tf.lite.TFLiteConverter.from_keras_model_file(saved_model_dir)
#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
