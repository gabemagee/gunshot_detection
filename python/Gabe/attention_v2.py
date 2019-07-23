from keras import backend as K
from keras.layers import Input, Dense, multiply, Flatten, Dropout, Lambda, normalization, Concatenate, Reshape, noise
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import BatchNormalization, MaxPool2D, Dropout, GlobalMaxPool2D
import numpy as np
from tensorflow.python.client import device_lib



image_size = 128
maxpool_size = (3,3)
drop_out_rate = 0.1
flattend_input_dims = image_size*image_size

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
# In[8]:

print("available gpus:",get_available_gpus())


os.environ["CUDA_VISIBLE_DEVICES"]="1"


#get a 128,128 attention tensor

flatten = Flatten()
attention_dense = Dense(flattend_input_dims, activation='softmax', name='attention_probs')
unflatten = Reshape(target_shape = (image_size,image_size,1))

input = Input(shape=(image_size,image_size,1),name='input')
flattened_input = flatten(input)
attention_probs = attention_dense(flattened_input)
attention_mul = multiply([flattened_input, attention_probs], name='attention_mul')
input_tensor = unflatten(attention_mul)

print(unflatten.output_shape)

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


model = Model(input=input,output=output_tensor)
model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=0.001, decay=0.001 / 100), metrics=['accuracy'])
model.summary()

base_directory = "/home/gamagee/workspace/gunshot_detection/test_train/"
X_train, X_test = np.load(base_directory+"X_train.npy").reshape((16294, 128, 128, 1)),np.load(base_directory+"X_test.npy").reshape((16294, 128, 128, 1))
Y_train, Y_test = np.load(base_directory+"y_train.npy"),np.load(base_directory+"y_test.npy")
model_history = model.fit(X_train, Y_train,batch_size=200,validation_data=(X_test,Y_test),nb_epoch=50)
