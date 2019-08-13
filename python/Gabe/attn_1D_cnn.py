import os

# ### Math Libraries

# In[ ]:


import numpy as np

# ### Data Pre-Processing Libraries

# In[ ]:


from array import array
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelBinarizer

# ### Deep Learning Libraries

# In[ ]:


import tensorflow as tf
from tensorflow.keras import Input, layers, optimizers, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


number_of_epochs = 100
batch_size = 32
optimizer = optimizers.Adam(lr=0.001, decay=0.001 / 100)
input_tensor = Input(shape=(44100, ))

attention_probs = Dense(44100,
                        activation='softmax', name='attention_probs')(input_tensor)
attention_mul = multiply(
    [input_tensor, attention_probs], name='attention_mul')

# ## Configuration of GPU for training (optional)

# In[ ]:


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# K.set_session(session)

# ## Model Architecture

# In[ ]:


x = layers.Conv1D(16, 9, activation="relu", padding="same")(input_tensor)
x = layers.Conv1D(16, 9, activation="relu", padding="same")(x)
x = layers.MaxPool1D(16)(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPool1D(4)(x)
x = layers.Dropout(rate=0.25)(x)

x = layers.Conv1D(256, 3, activation="relu", padding="same")(x)
x = layers.Conv1D(256, 3, activation="relu", padding="same")(x)
x = layers.GlobalMaxPool1D()(x)
x = layers.Dropout(rate=(0.5))(x)  # Increasing drop-out rate here to prevent overfitting

x = layers.Dense(64, activation="relu")(x)
x = layers.Dense(1028, activation="relu")(x)
output_tensor = layers.Dense(2, activation="softmax")(x)

model = tf.keras.Model(input_tensor, output_tensor)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=[auc, "accuracy"])

# ## Configuring model properties

# In[ ]:


model_filename = BASE_DIRECTORY + "gunshot_sound_model.pkl"

model_callbacks = [
    EarlyStopping(monitor='val_acc',
                  patience=15,
                  verbose=1,
                  mode='max'),

    ModelCheckpoint(model_filename, monitor='val_acc',
                    verbose=1,
                    save_best_only=True,
                    mode='max'),
]

# ### Debugging of the model's architecture (optional)

# In[ ]:


print(model.summary())

# ## Training & caching the model

# In[ ]:


History = model.fit(train_wav, train_label,
                    validation_data=[test_wav, test_label],
                    epochs=number_of_epochs,
                    callbacks=model_callbacks,
                    verbose=1,
                    batch_size=batch_size,
                    sample_weight=train_weights,
                    shuffle=True)
