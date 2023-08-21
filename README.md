# Speaker Audio Recognition

This repository contains two main scripts for training and testing speaker audio recognition models using either Convolutional Neural Networks (CNN) or Support Vector Machines (SVM). The models are trained on audio files and utilize Mel Frequency Cepstral Coefficient (MFCC) features.

## Files Description

### `train.py`

The `train.py` script is responsible for training the models. It defines a custom dataset class (`SpeakerAudioDataset`) for loading and preprocessing audio data, and a simple 1D CNN model (`VanillaCNN`) for classification. It also includes training and validation loops with performance metrics and early stopping. The training process can be performed with either CNN or SVM.

#### Usage

For CNN:
```bash
python train.py --model_type CNN --epochs 10
```
For SVM:
```bash
python train.py --model_type SVM
```

### `test.py`

The `test.py` script is used to test the trained models on a test dataset. It includes functions to test both the CNN and SVM models and print the true and predicted class labels for a specific audio sample.

#### Usage

For CNN:
```bash
python test.py --model_type CNN --data_index 0
```
For SVM:
```bash
python test.py --model_type SVM --data_index 0
```

## Dataset Structure

The dataset should be organized into folders corresponding to different classes, with audio files inside each folder. For example:

```bash
Dataset/
├── Train/
│   ├── Timm/
│   ├── Shenna/
│   └── John/
└── Test/
    ├── Timm/
    ├── Shenna/
    └── John/
```

## Requirements
- Python 3.x
- PyTorch
- torchaudio
- scikit-learn
- joblib
- seaborn
- matplotlib