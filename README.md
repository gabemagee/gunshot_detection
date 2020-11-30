# Gunshot Detection using Raspberry Pi

### How it works
Our  pipeline,  orchestrated  with  Python,  operates  with three  concurrent  threads:  one  to  continuously  capture  audio received  from  an  attached  microphone  and  put  two  seconds worth  of  said  audio  onto  an  audio  analysis  queue;  one  to analyze sound samples retrieved from the audio analysis queue and  verify  whether  or  not  a  gunshot  occurred  in  a  given sample;  and  finally  one  to  dispatch  an  SMS  alert  message to  a  predetermined  list  of  phone  numbers  if  a  gunshot  was detected in the segment of audio most recently analyzed. 

### Hardware
Our short message service (SMS) pipeline for detecting gunshots was deployed on a Raspberry Pi 3 Model B+ connected to an AT&T USBConnect Lightning Quickstart SMS modem as well as a Sizheng omnidirectional USB microphone. 

### Models
We trained three different models on a set of nearly 60,000 two-second audio samples to distinguish gunshots from other noises. The first model was a one-dimensional convolutional neural network that takes a two-second time sequence of audio as input. The second and third models were two-dimensional convolutional neural network that takes a spectrogram of a two-second audio sample as input. For inference in practice, the decision on a sample's class was reached by majority-rules consensus between the three models.

### Dataset
Our training features and samples along with our trained models can be found on [Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/2KI6IH).

### Citation
This repository contains the source code accompanying the paper [Low Cost Gunshot Detection using Deep Learning on the Raspberry Pi](https://ieeexplore.ieee.org/abstract/document/9006456).

Please cite as:
```
@INPROCEEDINGS{9006456,
  author={A. {Morehead} and L. {Ogden} and G. {Magee} and R. {Hosler} and B. {White} and G. {Mohler}},
  booktitle={2019 IEEE International Conference on Big Data (Big Data)}, 
  title={Low Cost Gunshot Detection using Deep Learning on the Raspberry Pi}, 
  year={2019},
  volume={},
  number={},
  pages={3038-3044},
  doi={10.1109/BigData47090.2019.9006456}}
 ```
