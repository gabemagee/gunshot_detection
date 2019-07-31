import tensorflow as tf
from tensorflow.keras import Input, layers, optimizers, backend as K
import os

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


saved_model_dir = "/home/gamagee/workspace/gunshot_detection/REU_Data/ryan_model/tflite_models/"
model_directory = "/home/gamagee/workspace/gunshot_detection/REU_Data/ryan_model/models/"


for model_filename in os.listdir(model_directory):
    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_directory+model_filename,custom_objects={"auc":auc})
    tflite_model = converter.convert()
    open(saved_model_dir+model_filename.split(".")[0]+".tflite", "wb").write(tflite_model)
