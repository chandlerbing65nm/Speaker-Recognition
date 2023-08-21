import argparse
import torchaudio
import torch
import os
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.svm import SVC
import torch.nn.functional as F

from train import VanillaCNN, SpeakerAudioDataset

def test_cnn(file_path, model_path, classes, index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VanillaCNN(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    dataset = SpeakerAudioDataset(file_path)
    sample, label = dataset[index]
    sample = sample.to(device)
    outputs = model(sample.unsqueeze(0))
    predicted_class_idx = torch.argmax(outputs, dim=1).item()

    print(f"True class: {classes[label]}, Predicted class: {classes[predicted_class_idx]}")

def test_svm(file_path, model_path, scaler_path, classes, index=0):
    clf = load(model_path)
    scaler = load(scaler_path)

    dataset = SpeakerAudioDataset(file_path)
    sample, label = dataset[index]
    sample_flatten = sample.numpy().flatten()
    sample_scaled = scaler.transform([sample_flatten])

    predicted_class_idx = clf.predict(sample_scaled)[0]

    print(f"True class: {classes[label]}, Predicted class: {classes[predicted_class_idx]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained model for speaker audio recognition")
    parser.add_argument("--file_path", type=str, help="Path to the audio file to test")
    parser.add_argument("--model_type", type=str, required=True, choices=["CNN", "SVM"], help="Model type to test ('CNN' or 'SVM')")
    parser.add_argument("--model_path", type=str, help="Path to the trained model")
    parser.add_argument("--scaler_path", type=str, help="Path to the trained scaler (only required for SVM)")
    parser.add_argument("--data_index", default=0, type=int)

    args = parser.parse_args()

    args.file_path = './Dataset/Test'
    args.model_path = f'Checkpoints/{args.model_type}_best_model'
    args.scaler_path = f'Checkpoints/{args.model_type}_scaler.joblib'

    classes = ['Timm', 'Shenna', 'John'] # Update this if your classes are different

    if args.model_type == "CNN":
        test_cnn(args.file_path, args.model_path+'.pth', classes, index=args.data_index)
    elif args.model_type == "SVM":
        if args.scaler_path is None:
            raise ValueError("scaler_path must be provided for SVM")
        test_svm(args.file_path, args.model_path+'.joblib', args.scaler_path, classes, index=args.data_index)