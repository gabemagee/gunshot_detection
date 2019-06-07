# gunshot_detection
Building a model to detect gunshots from audio and then put onto arduinos


# Steps
## 1. Build a working gunshot model
## 2. Use Post-Training Quantization to make it smaller
## 3. See if accuracy loss is acceptable. If not, return to 1
## 4. Convert the model to a TensorFlow Lite FlatBuffer
## 5. Convert the FlatBuffer to a C byte array
## 6. Integrate the TensorFlow Lite for Microcontrollers C++ library
## 7. Deploy to the arduino
## 8. Test in real world scenario
