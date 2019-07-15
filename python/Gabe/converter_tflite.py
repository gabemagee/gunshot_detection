import tensorflow as tf
from tensorflow.keras import Input, layers, optimizers, backend as K
import os

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


saved_model_dir = os.getcwd()+"/raspberry_pi/models/gunshot_sound_model_spectrograph_weighted_spectrogram.h5"
converter = tf.lite.TFLiteConverter.from_keras_model_file(saved_model_dir)
#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open(os.getcwd()+"/raspberry_pi/models/converted_model_weights.tflite", "wb").write(tflite_model)
