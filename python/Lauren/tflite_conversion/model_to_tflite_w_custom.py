#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#Turns a .h5 model file using custom metric into a TFLite flat buffer 


#see
    #https://www.tensorflow.org/lite/convert/python_api
    #https://www.tensorflow.org/lite/performance/post_training_quantization
    #https://www.tensorflow.org/lite/microcontrollers/build_convert#operation-support
    #https://www.tensorflow.org/versions/r1.14/api_docs/python/tf/lite/TFLiteConverter
    #https://github.com/keras-team/keras/issues/10104

#imports

import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.lite
from tensorflow.keras import backend as K


#print("version of tensorflow:")
#print(tf.__version__)


# custom metric: auc
def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


model_name = "/home/lauogden/models/gunshot_sound_model.h5"

converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name, custom_objects={'auc': auc})
print("\n made the converter using from_keras_model \n")

#post-training quantization
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
print(" did post training quantization \n")

#convert
tflite_model = converter.convert()
print(" converted successfully \n")

#save
open("/home/lauogden/models/converted_gunshot_model_ptq.tflite", "wb").write(tflite_model)
print(" saved successfully \n")


