# gunshot_detection
Building a model to detect gunshots from audio and then put onto arduinos


# Steps
### 1. Build a working gunshot model
Input: audio.
Output: Boolean yes/no
### 2. Use Post-Training Quantization to make the model smaller
### 3. See if accuracy loss is acceptable. If not, return to 1.
### 4. Convert the model to a TensorFlow Lite FlatBuffer
Use the TensorFlow Lite converter Python API: https://www.tensorflow.org/lite/convert/python_api
### 5. Convert the FlatBuffer to a C byte array
https://www.tensorflow.org/lite/microcontrollers/build_convert
### 6. Integrate the TensorFlow Lite for Microcontrollers C++ library
### 7. Deploy to the arduino
### 8. Test in real world scenario. If not acceptable, return to 1.
