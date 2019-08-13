from keras import backend as K
from keras.layers import Input, Dense, multiply, Flatten, Dropout, Lambda, normalization, Concatenate, Reshape, noise,BatchNormalization, MaxPool2D, Dropout, GlobalMaxPool2D, MaxPool1D, GlobalMaxPool1D
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv1D,AveragePool1D
import numpy as np
from tensorflow.python.client import device_lib
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard


maxpool_size = (3,3)
drop_out_rate = 0.1

base_directory = "/home/gamagee/workspace/gunshot_detection/test_train/"
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
# In[8]:

print("available gpus:",get_available_gpus())


from keras import backend as K
from keras.layers import Layer

class SmoothingLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = AveragePool1D(9)(x)
        a = []
        for i in x:
            b =
        return x

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)





#get a 128,128 attention tensor
def build_1D_cnn():
    input = Input(shape=(44100,1),name='input')
    flattened_input = Flatten()(input)
    attention_dense = Dense(4900, activation='softmax', name='attention_dense')
    attention_probs = attention_dense(flattened_input)

    attention_probs =

    attention_mul = multiply([flattened_input, attention_probs], name='attention_mul')
    input_tensor = Reshape(target_shape=(44100,1))(attention_mul)
    x = Conv1D(16, 3, activation="relu", padding="same")(input_tensor)
    x = MaxPool1D(3)(x)
    x = Dropout(rate=drop_out_rate)(x)

    x = Conv1D(32, 3, activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = MaxPool1D(3)(x)
    x = Dropout(rate=drop_out_rate)(x)

    x = Conv1D(64, 3, activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = MaxPool1D(3)(x)
    x = Dropout(rate=drop_out_rate)(x)

    x = Conv1D(256, 3, activation="relu", padding="same")(x)
    #x = BatchNormalization()(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(rate=(drop_out_rate * 2))(x) # Increasing drop-out rate here to prevent overfitting

    x = Dense(64, activation="relu")(x)
    x = Dense(1028, activation="relu")(x)
    output_tensor = Dense(2, activation="softmax")
    output = output_tensor(x)
    print(output_tensor.output_shape)
    model = Model(input=input,output=output, name= "1D")

    return model


def build_2D_cnn():
    image_size = 128
    flattend_input_dims = image_size*image_size
    flatten = Flatten()
    attention_dense = Dense(flattend_input_dims, activation='softmax', name='attention_probs')
    unflatten = Reshape(target_shape = (image_size,image_size,1))

    input = Input(shape=(image_size,image_size,1),name='input')
    flattened_input = flatten(input)
    attention_probs = attention_dense(flattened_input)
    attention_mul = multiply([flattened_input, attention_probs], name='attention_mul')
    input_tensor = unflatten(attention_mul)

    x = Conv2D(16, (3,3), activation="relu", padding="same")(input_tensor)
    x = BatchNormalization()(x)
    x = MaxPool2D(maxpool_size)(x)
    x = Dropout(rate=drop_out_rate)(x)

    x = Conv2D(32, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(maxpool_size)(x)
    x = Dropout(rate=drop_out_rate)(x)

    x = Conv2D(64, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(maxpool_size)(x)
    x = Dropout(rate=drop_out_rate)(x)

    x = Conv2D(256, (3,3), activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    x = GlobalMaxPool2D()(x)
    x = Dropout(rate=(drop_out_rate * 2))(x) # Increasing drop-out rate here to prevent overfitting

    x = Dense(64, activation="relu")(x)
    x = Dense(1028, activation="relu")(x)
    output_tensor = Dense(2, activation="softmax")(x)


    model = Model(input=input,output=output_tensor, name= "2D")

    return model

model = build_2D_cnn()
#model = build_1D_cnn()

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=0.0001, decay=0.001 / 100), metrics=['accuracy'])
model.summary()

model_filename = base_directory + "attention_model_gabe.pkl"

model_callbacks = [
        EarlyStopping(monitor='val_acc',
                      patience=15,
                      verbose=1,
                      mode='max'),

        ModelCheckpoint(model_filename, monitor='val_acc',
                        verbose=1,
                        save_best_only=True,
                        mode='max'),]

data_dir = "/home/gamagee/workspace/gunshot_detection/REU_Data/spectrogram_training/samples_and_labels/attn/"

X_train, X_test = np.load(base_directory+"X_train.npy").reshape((16294, 128, 128, 1)),np.load(base_directory+"X_test.npy").reshape((16294, 128, 128, 1))
#X_train, X_test = np.load(data_dir+"X_train.npy").reshape(16294,44100,1),np.load(data_dir+"X_test.npy").reshape(16294,44100,1)

Y_train, Y_test = np.load(base_directory+"y_train.npy"),np.load(base_directory+"y_test.npy")

print(Y_train.shape)

model_history = model.fit(X_train, Y_train,batch_size=64,validation_data=(X_test,Y_test),callbacks=model_callbacks,epochs=50)

model_save_file = base_directory + "attention_model_gabe.h5"

model.save(model_save_file)
print("saved at: ",model_save_file)
