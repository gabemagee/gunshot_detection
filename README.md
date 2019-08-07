# Gunshot Detection using Raspberry Pi

### How it works
Our  pipeline,  orchestrated  with  Python,  operates  with three  concurrent  threads:  one  to  continuously  capture  audio received  from  an  attached  microphone  and  put  two  secondsworth  of  said  audio  onto  an  audio  analysis  queue;  one  to analyze sound samples retrieved from the audio analysis queueand  verify  whether  or  not  a  gunshot  occurred  in  a  given sample;  and  finally  one  to  dispatch  an  SMS  alert  messageto  a  predetermined  list  of  phone  numbers  if  a  gunshot  was detected in the segment of audio most recently analyzed. 

### Hardware
Our short message service (SMS) pipeline for detecting gun-shots was deployed on a Raspberry Pi 3 Model B+ connected to an AT&T USBConnect Lightning Quickstart SMS modem as  well  as  a  Sizheng  omnidirectional  USB  microphone. 

### Models

We trained three different models on a set of 50,000 2-second audio samples to distinguish gunshots from other noises. The first model was a 1-dimensional Convolutional Neural Network that takes a 2-second time sequence of audio as input. The second and third models were 2-dimensional Convolutional Neural Network that takes a spectrogram of a 2-second audio sample as input. For inference in practice, the decision on a sample was reached by majority-rules consensus between the three.
