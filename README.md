# Brain Tumor MRI Classifier

This project is an engineering split of the original notebook
`brain-tumor-mri-accuracy-99.ipynb`. The model logic is kept the same:
Xception backbone, Flatten, Dropout, Dense, Dropout, and final Softmax
classification head.

## Project Structure

```text
.
├── dataset.py
├── model.py
├── predict.py
├── train.py
├── utils.py
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Dataset Layout

The code expects the same directory layout used in the notebook:

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

## Train

Default paths match the Kaggle notebook paths:

```bash
python train.py
```

For local data:

```bash
python train.py \
  --train-dir /path/to/brain-tumor-mri-dataset/Training \
  --test-dir /path/to/brain-tumor-mri-dataset/Testing
```

Useful options:

```bash
python train.py --epochs 10 --batch-size 32 --output-dir artifacts
python train.py --no-plots
```

Training writes:

```text
artifacts/
├── brain_tumor_xception.keras
└── class_indices.json
```

## Predict

```bash
python predict.py \
  --image /path/to/image.jpg \
  --model artifacts/brain_tumor_xception.keras \
  --class-indices artifacts/class_indices.json
```

To print probabilities without opening a plot window:

```bash
python predict.py --image /path/to/image.jpg --no-show
```

## Notes

- `model.py` contains only the original Xception-based model definition.
- `dataset.py` contains dataframe creation, train/validation split, and
  `ImageDataGenerator` setup.
- `train.py` runs the original training and evaluation flow.
- `predict.py` contains the single-image prediction flow from the notebook.
- `utils.py` contains plotting, evaluation, and class-index helpers.
