import tensorflow as tf
from tensorflow.keras import backend as K

BASE_DIRECTORY = "/home/alexm/Downloads/"

# ROC (AUC) metric - Uses the import "from tensorflow.keras import backend as K"


def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


converter = tf.lite.TFLiteConverter.from_keras_model_file(
    "/home/alexm/Downloads/128_128_RYAN_spectrogram_model.h5", custom_objects={"auc": auc})
converter.post_training_quantize = True
tflite_model = converter.convert()
open(BASE_DIRECTORY + "128_x_128_2D.tflite", "wb").write(tflite_model)
