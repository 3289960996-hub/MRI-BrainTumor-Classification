# MRI Brain Tumor Classification

[![CI](https://github.com/3289960996-hub/MRI-BrainTumor-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/3289960996-hub/MRI-BrainTumor-Classification/actions/workflows/ci.yml)

Production-style Python project for MRI brain tumor image classification,
refactored from the Kaggle notebook `brain-tumor-mri-accuracy-99.ipynb`.

The model algorithm is intentionally unchanged from the notebook:

- ImageNet-pretrained `Xception`
- `Flatten`
- `Dropout(0.3)`
- `Dense(128, activation="relu")`
- `Dropout(0.25)`
- `Dense(4, activation="softmax")`
- `Adamax(learning_rate=0.001)`
- `categorical_crossentropy`
- metrics: accuracy, precision, recall

This repository focuses on engineering the notebook into a runnable machine
learning project without redesigning the model.

## Project Structure

```text
.
├── .github/workflows/ci.yml
├── artifacts/
│   └── .gitkeep
├── configs/
│   └── default.yaml
├── data/
│   └── .gitkeep
├── tests/
│   ├── test_cli.py
│   └── test_config.py
├── config.py
├── dataset.py
├── model.py
├── predict.py
├── train.py
├── utils.py
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
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

The code supports common image extensions:

```text
.jpg, .jpeg, .png, .bmp, .tif, .tiff
```

The original Kaggle paths are stored in `configs/default.yaml`:

```yaml
data:
  train_dir: /kaggle/input/brain-tumor-mri-dataset/Training
  test_dir: /kaggle/input/brain-tumor-mri-dataset/Testing
```

For local training, pass your own dataset paths with command-line arguments.

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

For tests and CI-like checks:

```powershell
pip install -r requirements-dev.txt
```

## Configuration

Default training settings live in:

```text
configs/default.yaml
```

Current defaults match the original notebook:

```yaml
training:
  epochs: 10
  batch_size: 32
  img_size: [299, 299]
```

You can either edit the YAML file or override values from the command line.

## Training

Using defaults from `configs/default.yaml`:

```powershell
python train.py
```

Using a custom config:

```powershell
python train.py --config configs/default.yaml
```

Using local Windows dataset paths:

```powershell
python train.py `
  --train-dir "C:\path\to\brain-tumor-mri-dataset\Training" `
  --test-dir "C:\path\to\brain-tumor-mri-dataset\Testing"
```

Useful overrides:

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

After training, predict a single MRI image:

```powershell
python predict.py `
  --image "C:\path\to\image.jpg" `
  --model artifacts\brain_tumor_xception.keras `
  --class-indices artifacts\class_indices.json
```

If you use the default artifact paths from `configs/default.yaml`, this is
enough:

```powershell
python predict.py --image "C:\path\to\image.jpg"
```

Print probabilities without opening a plot window:

```powershell
python predict.py --image "C:\path\to\image.jpg" --no-show
```

## Testing

Run lightweight tests:

```powershell
pytest -q
```

The tests intentionally avoid full training because that requires the MRI
dataset and TensorFlow runtime. CI checks syntax, CLI entry points, and config
helpers.

## CI

GitHub Actions is configured in:

```text
.github/workflows/ci.yml
```

It runs on push and pull request to `main`:

- Python setup
- lightweight dependency installation
- syntax compilation
- pytest checks

## File Responsibilities

- `config.py`: YAML config loading and nested config lookup.
- `dataset.py`: dataset validation, dataframe creation, validation/test split,
  and Keras `ImageDataGenerator` setup.
- `model.py`: original Xception model definition and compilation.
- `train.py`: runnable training entry point with fit, validation, evaluation,
  plots, and artifact saving.
- `predict.py`: runnable single-image prediction entry point.
- `utils.py`: plotting, evaluation, serialization, and helper utilities.

## Medical Disclaimer

This project is for learning and software engineering demonstration only. It is
not a medical device and must not be used for clinical diagnosis or treatment
decisions.

See `MODEL_CARD.md` for intended use, limitations, and evaluation notes.

## License

MIT License. See `LICENSE`.
