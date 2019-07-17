import tensorflow as tf
from tensorflow.keras import Input, layers, optimizers, backend as K
import os

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


saved_model_dir = "/home/gamagee/workspace/gunshot_detection/REU_Data/spectrogram_training/models/"
model = saved_model_dir+"spectrogram_gunshot_model_1.h5"
converter = tf.lite.TFLiteConverter.from_keras_model_file(model)
#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open(saved_model_dir+"spectrogram_gunshot_model_1.tflite", "wb").write(tflite_model)
print("done 1")

model = saved_model_dir+"gunshot_sound_model_1d.h5"
converter = tf.lite.TFLiteConverter.from_keras_model_file(model,custom_objects={"auc":auc})
#converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open(saved_model_dir+"gunshot_sound_model_1d.tflite", "wb").write(tflite_model)
