# MRI Brain Tumor Classification

This repository is a runnable Python project refactored from the Kaggle notebook
`brain-tumor-mri-accuracy-99.ipynb`.

The model logic is preserved from the original notebook. The project uses an
ImageNet-pretrained Xception backbone followed by Flatten, Dropout, Dense,
Dropout, and a 4-class Softmax output layer.

## Features

- Load MRI image folders into Pandas dataframes
- Split the original testing set into validation and test subsets
- Apply the same Keras `ImageDataGenerator` preprocessing used in the notebook
- Train the original Xception-based classifier
- Evaluate train, validation, and test metrics
- Plot training curves and confusion matrix
- Save trained model, class indices, and training history
- Predict a single MRI image from the command line

## Project Structure

```text
.
├── dataset.py
├── model.py
├── train.py
├── predict.py
├── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

Expected dataset layout:

```text
brain-tumor-mri-dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The original notebook paths are used as defaults:

```text
/kaggle/input/brain-tumor-mri-dataset/Training
/kaggle/input/brain-tumor-mri-dataset/Testing
```

For local Windows usage, pass your local dataset paths through command-line
arguments.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Training

Run with Kaggle-style default paths:

```powershell
python train.py
```

Run with local dataset paths:

```powershell
python train.py `
  --train-dir "C:\path\to\brain-tumor-mri-dataset\Training" `
  --test-dir "C:\path\to\brain-tumor-mri-dataset\Testing"
```

Useful options:

```powershell
python train.py --epochs 10 --batch-size 32 --output-dir artifacts
python train.py --no-plots
```

Training outputs:

```text
artifacts/
├── brain_tumor_xception.keras
├── class_indices.json
└── training_history.json
```

## Prediction

After training, predict one MRI image:

```powershell
python predict.py `
  --image "C:\path\to\image.jpg" `
  --model artifacts\brain_tumor_xception.keras `
  --class-indices artifacts\class_indices.json
```

Print probabilities without opening the plot window:

```powershell
python predict.py --image "C:\path\to\image.jpg" --no-show
```

## File Responsibilities

- `dataset.py`: dataset folder validation, dataframe creation, validation/test
  split, and Keras image generator creation.
- `model.py`: original Xception-based Keras model definition and compilation.
- `train.py`: command-line training entry point with fit, validation, metrics,
  plots, and artifact saving.
- `predict.py`: command-line single-image prediction.
- `utils.py`: plotting, evaluation, class-index serialization, and training
  history helpers.

## Notes

- The model architecture and training logic are intentionally kept aligned with
  the original notebook.
- Large generated files such as trained models and datasets are ignored by Git.
- If TensorFlow cannot use GPU on your machine, it will still run on CPU but
  training will be slower.
