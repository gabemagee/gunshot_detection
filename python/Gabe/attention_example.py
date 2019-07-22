'''
2016 by Jacob Zweig @jacobzweig
build RCNN networks in keras
'''
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import add, Convolution2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD


def makeModel(nbChannels, shape1, shape2, nbClasses, nbRCL=5,nbFilters=128, filtersize = 3):

	model = BuildRCNN(nbChannels, shape1, shape2, nbClasses, nbRCL, nbFilters, filtersize)
	return model

def BuildRCNN(nbChannels, shape1, shape2, nbClasses, nbRCL, nbFilters, filtersize):

    def RCL_block(l_settings, l, pool=True, increase_dim=False):
        input_num_filters = l_settings.output_shape[1]
        if increase_dim:
            out_num_filters = input_num_filters*2
        else:
            out_num_filters = input_num_filters

        conv1 = Convolution2D(out_num_filters, 1, 1, border_mode='same')
        stack1 = conv1(l)
        stack2 = BatchNormalization()(stack1)
        stack3 = PReLU()(stack2)

        conv2 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', init = 'he_normal')
        stack4 = conv2(stack3)
        stack5 = add([stack1, stack4])
        stack6 = BatchNormalization()(stack5)
        stack7 = PReLU()(stack6)

        conv3 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
        stack8 = conv3(stack7)
        stack9 = add([stack1, stack8])
        stack10 = BatchNormalization()(stack9)
        stack11 = PReLU()(stack10)

        conv4 = Convolution2D(out_num_filters, filtersize, filtersize, border_mode='same', weights = conv2.get_weights())
        stack12 = conv4(stack11)
        stack13 = add([stack1, stack12])
        stack14 = BatchNormalization()(stack13)
        stack15 = PReLU()(stack14)

        if pool:
            stack16 = MaxPooling2D((2, 2), border_mode='same')(stack15)
            stack17 = Dropout(0.1)(stack16)
        else:
            stack17 = Dropout(0.1)(stack15)

        return stack17

    #Build Network
    input_img = Input(shape=(nbChannels, shape1, shape2))
    conv_l = Convolution2D(nbFilters, filtersize, filtersize, border_mode='same', activation='relu')
    l = conv_l(input_img)

    for n in range(nbRCL):
        if n % 2 ==0:
            l = RCL_block(conv_l, l, pool=False)
        else:
            l = RCL_block(conv_l, l, pool=True)

    out = Flatten()(l)
    l_out = Dense(nbClasses, activation = 'softmax')(out)

    model = Model(input = input_img, output = l_out)

    return model

import numpy as np
from keras.utils.np_utils import to_categorical
import librosa
X_train = np.load("X_train.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_test = np.load("y_test.npy")

image_size = 128
X_train.shape = (len(X_train),image_size,image_size,1)
X_test.shape = (len(X_test),image_size,image_size,1)


y_trainCAT = to_categorical(y_train)
y_testCAT = to_categorical(y_test)
print("SHAPE:",X_train.shape)
model = makeModel(1,128,128,2)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.5, nesterov=True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])

print(model.summary)

model_history = model.fit(X_train, y_trainCAT,batch_size=1,validation_data=(X_test,y_testCAT),nb_epoch=1)
