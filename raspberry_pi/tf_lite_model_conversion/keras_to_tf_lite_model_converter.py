import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras import Input, layers, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("The version of TensorFlow installed:", tf.__version__)
print("The version of TensorFlow.Keras installed:", keras.__version__)

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

# ## Model Architecture
drop_out_rate = 0.1
learning_rate = 0.001
number_of_epochs = 100
number_of_classes = 2
batch_size = 32
optimizer = optimizers.Adam(learning_rate, learning_rate / 100)
input_shape = (44100, 1)
input_tensor = Input(shape = input_shape)
metrics = [auc, "accuracy"]

x = layers.Conv1D(16, 9, activation = "relu", padding = "same")(input_tensor)
x = layers.Conv1D(16, 9, activation = "relu", padding = "same")(x)
x = layers.MaxPool1D(16)(x)
x = layers.Dropout(rate = drop_out_rate)(x)

x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate = drop_out_rate)(x)

x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
x = layers.Conv1D(32, 3, activation = "relu", padding = "same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate = drop_out_rate)(x)

x = layers.Conv1D(256, 3, activation = "relu", padding = "same")(x)
x = layers.Conv1D(256, 3, activation = "relu", padding = "same")(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dropout(rate = (drop_out_rate * 2))(x) # Increasing drop-out rate here to prevent overfitting

x = layers.Dense(64, activation = "relu")(x)
x = layers.Dense(1028, activation = "relu")(x)
output_tensor = layers.Dense(number_of_classes, activation = "softmax")(x)

model = tf.keras.Model(input_tensor, output_tensor)
model.compile(optimizer = optimizer, loss = keras.losses.binary_crossentropy, metrics = metrics)

# # Converting the model to a protocol buffer (.pb) file
dependencies = {
     "auc": auc
}
# model = keras.models.load_model("gunshot_sound_model.h5", custom_objects = dependencies)
# K.set_learning_phase(0)  # 0 for testing mode, 1 for training mode

# frozen_graph = freeze_session(K.get_session(),
#                               output_names=[out.op.name for out in model.outputs])
# tf.io.write_graph(frozen_graph, "~/Downloads", "gunshot_sound_model.pb", as_text = False)

# Converting the model to a flat buffer (.tflite) file
converter = tf.lite.TFLiteConverter.from_keras_model_file("../models/gunshot_sound_model.h5", custom_objects = dependencies)
tflite_model = converter.convert()
open("../models/gunshot_sound_model.tflite", "wb").write(tflite_model)
