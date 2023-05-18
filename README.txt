


# Auxiliary-Attention-Pooling-Network-Based-Recording-Device-Detection-System

## Description

This project "Auxiliary Attention Pooling Network-Based Recording Device Detection System" aims to develop a system that can detect and identify the recording device used to capture an audio file. With the increasing variety and quality of microphones and recording devices available, it has become challenging to determine the source of an audio recording based solely on its characteristics.

The motivation behind this project is to address the need for accurately identifying recording devices in various applications. This identification can have significant implications in areas such as audio forensics, speech-to-text conversion, authentication of audio files in legal proceedings, and detecting audio bots.

## Main features

- Utilizes Multitasking based Auxiliary Attention Pooling Network for audio device classification.
- Implements model optimization techniques including white noise augmentation, kernel size optimization, and loss weightage optimization.
- Presents experimental results comparing validation and test accuracy of different models and techniques.
- Selects top three models based on validation accuracy for the final system.
- Achieves an accuracy of 86.9% on test data, surpassing previously published results by 2.9%.

## Tech Stack and concepts used

- Python
- Keras
- OpenCV
- Scikit-learn
- Convolution Neural Network
- Long Short term Network
- Signal Processing
- Audio Sampling
- GPU Acceleration
- White Noise Augmentation
- MFCC , Feature Bank , Log features Bank

## Thought behind the project

The key idea behind the project is to leverage the unique acoustic characteristics and imperfections introduced by different recording devices. Every device has its own distinctive noise profile, frequency response, and signal processing characteristics that influence the captured audio. By extracting and analyzing these device-specific features, it becomes possible to differentiate between recordings made by different devices.

The project proposes using an attention-based pooling network in combination with Convolutional Neural Networks (CNNs) and an auxiliary task. Attention mechanisms allow the model to focus on specific portions of the audio signal that are most relevant for device identification. The pooling operation helps capture the salient information across multiple layers of the network, enhancing the discriminative power.

The auxiliary task is incorporated to further improve the model's performance. By training the model to perform an additional related task, such as audio classification or language identification, it learns more robust and discriminative representations that can be utilized for recording device detection.

## Datasets
###  MOBIPHONE DB

- 20 cell phones. 

- 24 speakers from TIMIT.

- 10 sentences from eache speaker.

- 4800 utterances.

| Model                                            | Validation Set Accuracy | Testing Set Accuracy |
| ------------------------------------------------- | --------------------- | -------------------- |
|Simple CNN + LSTM based network |73.4% |70%|
|Speaker based Cross Validation |68.6% |64.8%|
|Auxiliary Model Architecture |75.9% |77.7%|
|White Noise Augmentation| 77.38% |81.1%|
|Kernel size optimization |82.89%| 81.01%|
|Loss Weightage Optimization| 82.6% |80.2%|
|Best 3 out of 6 Model |87.5% |86.9%|