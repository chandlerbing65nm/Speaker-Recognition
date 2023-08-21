# Speaker Audio Recognition

This repository contains two main scripts for training and testing speaker audio recognition models using either Convolutional Neural Networks (CNN) or Support Vector Machines (SVM). The models are trained on audio files and utilize Mel Frequency Cepstral Coefficient (MFCC) features.

## Files Description

### `train.py`

The `train.py` script is responsible for training the models. It defines a custom dataset class (`SpeakerAudioDataset`) for loading and preprocessing audio data, and a simple 1D CNN model (`VanillaCNN`) for classification. It also includes training and validation loops with performance metrics and early stopping. The training process can be performed with either CNN or SVM.

#### Usage

```bash
python train.py --root_dir /path/to/dataset --model_type CNN --batch_size 2 --num_classes 3 --epochs 10
```

### `test.py`

The `test.py` script is used to test the trained models on a test dataset. It includes functions to test both the CNN and SVM models and print the true and predicted class labels for a specific audio sample.

#### Usage

For CNN:
```bash
python test.py --file_path /path/to/testdata --model_type CNN --model_path /path/to/model --data_index 0
```
For SVM:
```bash
python test.py --file_path /path/to/testdata --model_type SVM --model_path /path/to/model --scaler_path /path/to/scaler --data_index 0
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