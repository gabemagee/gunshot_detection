import numpy as np
import tensorflow as tf
from time import time

# Loads TFLite model and allocate tensors
interpreter = tf.lite.Interpreter("../models/spectro_no_variables.tflite")
interpreter.allocate_tensors()

# Gets input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Tests model on random input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time()
interpreter.invoke()
print("Inference time with Gabe's 2D CNN model:", time() - start_time)

# The function `get_tensor()` returns a copy of the tensor data
# Use `tensor()` in order to get a pointer to the tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Probabilities derived by Gabe's 2D CNN model:", output_data)


# Loads TFLite model and allocate tensors
interpreter = tf.lite.Interpreter("../models/gunshot_sound_model.tflite")
interpreter.allocate_tensors()

# Gets input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Tests model on random input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time()
interpreter.invoke()
print("Inference time with Alex's 1D CNN model:", time() - start_time)

# The function `get_tensor()` returns a copy of the tensor data
# Use `tensor()` in order to get a pointer to the tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Probabilities derived by Alex's 1D CNN model:", output_data)

# Loads TFLite model and allocate tensors
interpreter = tf.lite.Interpreter("../models/gunshot_2d_spectrogram_model.tflite")
interpreter.allocate_tensors()

# Gets input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Tests model on random input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time()
interpreter.invoke()
print("Inference time with Alex's 2D CNN model:", time() - start_time)

# The function `get_tensor()` returns a copy of the tensor data
# Use `tensor()` in order to get a pointer to the tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print("Probabilities derived by Alex's 2D CNN model:", output_data)
