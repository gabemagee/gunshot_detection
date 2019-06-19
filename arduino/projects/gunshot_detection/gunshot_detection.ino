/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Template sketch that calls into the detailed TensorFlow Lite codebase.

// Include an empty header so that Arduino knows to build the TF Lite library.
#include <TensorFlowLite.h>

extern int loadModel();
//extern int buildInterpreter();
//extern int requestMemoryInformation();
//extern int setupDataLoader();
//extern int setupCommandRecognizer();
//extern int analyzeMicrophoneAudio();

void setup() {
  // Initializes the audio analysis pipeline.
  Serial.println(loadModel());
//  Serial.println(buildInterpreter());
//  Serial.println(requestMemoryInformation());
//  Serial.println(setupDataLoader());
//  Serial.println(setupCommandRecognizer());
}

void loop() {
//  Serial.println(analogRead(0));
//  digitalWrite(LED_BUILTIN, HIGH);
//  delay(1000);
//  digitalWrite(LED_BUILTIN, LOW);
//  delay(1000);
//  Serial.println(analyzeMicrophoneAudio());
}
