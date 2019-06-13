#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#see
    #https://www.tensorflow.org/lite/convert/python_api
    #https://www.tensorflow.org/lite/performance/post_training_quantization
    #https://www.tensorflow.org/lite/microcontrollers/build_convert#operation-support

#imports

import tensorflow as tf


# In[ ]:


#use the saved model (.h5 file)

converter = tf.lite.TFLiteConverter.from_keras_model_file("/home/lauogden/data/gunshot_sound_model.h5")

#post-training quantization
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

#convert
tflite_model = converter.convert()

#save
open("/home/lauogden/data/converted_model.tflite", "wb").write(tflite_model)

