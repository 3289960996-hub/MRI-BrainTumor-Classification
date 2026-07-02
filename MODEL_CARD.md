# Model Card

## Model Details

- Task: Brain tumor MRI image classification
- Architecture: Xception transfer learning classifier
- Input size: 299 x 299 RGB image
- Output classes: glioma, meningioma, notumor, pituitary
- Framework: TensorFlow / Keras

The architecture and training choices are migrated from the original Kaggle
notebook without redesigning the model.

## Intended Use

This model is intended for:

- deep learning practice
- notebook-to-project refactoring practice
- machine learning engineering portfolio demonstration
- command-line training and inference examples

It is not intended for real clinical use.

## Training Data Format

The project expects images organized by class folder:

```text
Training/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/
```

The same class folder layout is expected under `Testing/`.

## Preprocessing

The preprocessing follows the original notebook:

- resize images to 299 x 299
- rescale pixel values by `1 / 255`
- apply brightness augmentation in the training and validation generators

## Evaluation

The training script reports:

- training loss and accuracy
- validation loss and accuracy
- test loss and accuracy
- precision
- recall
- confusion matrix
- classification report

The exact scores depend on dataset version, hardware, TensorFlow version, and
training randomness.

## Limitations

- The repository does not include the MRI dataset.
- The model has not been validated for clinical diagnosis.
- The original notebook split strategy is preserved, including splitting the
  provided testing set into validation and test subsets.
- Performance claims should be reproduced in the target environment before use.

## Ethical and Safety Notes

Medical image models can be sensitive to data distribution, scanner type,
preprocessing choices, and labeling quality. This project should be treated as
an educational example only.
