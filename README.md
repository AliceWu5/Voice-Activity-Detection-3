# Voice-Activity-Detection

# Introduction

Voice Activity Detection is a technique that detect whether human voice does or does not exist in sound signals. In this project, we use the Support Vector Machine model as a classifier. The data is extracted from videos on https://youtube.com/. 

# About the code

## Data

We use .wav extracted from videos on youtube and allocate them into two folders which are 'Data/Voice/' (including audio files that contain human voice) and 'Data/NonVoice/' (including audio files that do not contain human voice).

## Generate input

In this part, we extract MFCC features from .wav files and create a input matrix with the shape (#frames, #features). After that, we write them to 'voice.txt' and 'nonvoice.txt' for next use.

## Training data

In this part, we use SVM model to train 90% of data generated in 'voice.txt' and 'nonvoice.txt', and 10% left is use for testing.

## Testing

In this part, we calculate the precision and recall.

Author: Nguyen Quoc Huy
