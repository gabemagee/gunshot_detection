## Gunshot Detection using Raspberry Pi

### How it works
Our  pipeline,  orchestrated  with  Python,  operates  with three  concurrent  threads:  one  to  continuously  capture  audio received  from  an  attached  microphone  and  put  two  secondsworth  of  said  audio  onto  an  audio  analysis  queue;  one  to analyze sound samples retrieved from the audio analysis queueand  verify  whether  or  not  a  gunshot  occurred  in  a  given sample;  and  finally  one  to  dispatch  an  SMS  alert  messageto  a  predetermined  list  of  phone  numbers  if  a  gunshot  was detected in the segment of audio most recently analyzed. 
