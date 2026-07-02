import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def save_class_indices(class_indices, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(class_indices, f, ensure_ascii=False, indent=2)


def load_class_indices(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_class_counts(df, title="Count of images in each class"):
    plt.figure(figsize=(15, 7))
    ax = sns.countplot(data=df, y=df["Class"])
    plt.xlabel("")
    plt.ylabel("")
    plt.title(title, fontsize=20)
    ax.bar_label(ax.containers[0])
    plt.show()


def plot_samples(generator):
    class_dict = generator.class_indices
    classes = list(class_dict.keys())
    images, labels = next(generator)

    plt.figure(figsize=(20, 20))

    for i, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(image)
        class_name = classes[np.argmax(label)]
        plt.title(class_name, color="k", fontsize=15)

    plt.show()


def plot_training_history(hist):
    tr_acc = hist.history["accuracy"]
    tr_loss = hist.history["loss"]
    tr_per = hist.history["precision"]
    tr_recall = hist.history["recall"]
    val_acc = hist.history["val_accuracy"]
    val_loss = hist.history["val_loss"]
    val_per = hist.history["val_precision"]
    val_recall = hist.history["val_recall"]

    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    index_precision = np.argmax(val_per)
    per_highest = val_per[index_precision]
    index_recall = np.argmax(val_recall)
    recall_highest = val_recall[index_recall]

    epochs = [i + 1 for i in range(len(tr_acc))]
    loss_label = f"Best epoch = {str(index_loss + 1)}"
    acc_label = f"Best epoch = {str(index_acc + 1)}"
    per_label = f"Best epoch = {str(index_precision + 1)}"
    recall_label = f"Best epoch = {str(index_recall + 1)}"

    plt.figure(figsize=(20, 12))
    plt.style.use("fivethirtyeight")

    plt.subplot(2, 2, 1)
    plt.plot(epochs, tr_loss, "r", label="Training loss")
    plt.plot(epochs, val_loss, "g", label="Validation loss")
    plt.scatter(index_loss + 1, val_lowest, s=150, c="blue", label=loss_label)
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(epochs, tr_acc, "r", label="Training Accuracy")
    plt.plot(epochs, val_acc, "g", label="Validation Accuracy")
    plt.scatter(index_acc + 1, acc_highest, s=150, c="blue", label=acc_label)
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(epochs, tr_per, "r", label="Precision")
    plt.plot(epochs, val_per, "g", label="Validation Precision")
    plt.scatter(index_precision + 1, per_highest, s=150, c="blue", label=per_label)
    plt.title("Precision and Validation Precision")
    plt.xlabel("Epochs")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(epochs, tr_recall, "r", label="Recall")
    plt.plot(epochs, val_recall, "g", label="Validation Recall")
    plt.scatter(index_recall + 1, recall_highest, s=150, c="blue", label=recall_label)
    plt.title("Recall and Validation Recall")
    plt.xlabel("Epochs")
    plt.ylabel("Recall")
    plt.legend()
    plt.grid(True)

    plt.suptitle("Model Training Metrics Over Epochs", fontsize=16)
    plt.show()


def evaluate_model(model, tr_gen, valid_gen, ts_gen):
    train_score = model.evaluate(tr_gen, verbose=1)
    valid_score = model.evaluate(valid_gen, verbose=1)
    test_score = model.evaluate(ts_gen, verbose=1)

    print(f"Train Loss: {train_score[0]:.4f}")
    print(f"Train Accuracy: {train_score[1] * 100:.2f}%")
    print("-" * 20)
    print(f"Validation Loss: {valid_score[0]:.4f}")
    print(f"Validation Accuracy: {valid_score[1] * 100:.2f}%")
    print("-" * 20)
    print(f"Test Loss: {test_score[0]:.4f}")
    print(f"Test Accuracy: {test_score[1] * 100:.2f}%")

    return train_score, valid_score, test_score


def plot_confusion_matrix(model, ts_gen, class_dict):
    preds = model.predict(ts_gen)
    y_pred = np.argmax(preds, axis=1)
    cm = confusion_matrix(ts_gen.classes, y_pred)
    labels = list(class_dict.keys())

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Truth Label")
    plt.show()

    clr = classification_report(ts_gen.classes, y_pred)
    print(clr)
    return y_pred
