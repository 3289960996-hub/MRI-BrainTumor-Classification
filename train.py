import argparse
import os
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train brain tumor MRI classifier.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Training dataset directory.",
    )
    parser.add_argument(
        "--test-dir",
        default=None,
        help="Testing dataset directory.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--img-size", type=int, nargs=2, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--no-plots", action="store_true", help="Disable plots.")
    return parser.parse_args()


def main():
    args = parse_args()
    from config import get_nested, load_config

    from dataset import create_generators, load_datasets
    from model import build_model
    from utils import (
        ensure_dir,
        evaluate_model,
        plot_confusion_matrix,
        plot_training_history,
        save_class_indices,
        save_history,
    )

    cfg = load_config(args.config)
    train_dir = args.train_dir or get_nested(
        cfg, "data.train_dir", "/kaggle/input/brain-tumor-mri-dataset/Training"
    )
    test_dir = args.test_dir or get_nested(
        cfg, "data.test_dir", "/kaggle/input/brain-tumor-mri-dataset/Testing"
    )
    epochs = args.epochs or get_nested(cfg, "training.epochs", 10)
    batch_size = args.batch_size or get_nested(cfg, "training.batch_size", 32)
    img_size = tuple(args.img_size or get_nested(cfg, "training.img_size", [299, 299]))
    output_dir = args.output_dir or get_nested(cfg, "output.dir", "artifacts")
    model_name = args.model_name or get_nested(
        cfg, "output.model_name", "brain_tumor_xception.keras"
    )

    args.output_dir = output_dir
    ensure_dir(args.output_dir)

    tr_df, valid_df, ts_df = load_datasets(train_dir, test_dir)
    print(f"Training images: {len(tr_df)}")
    print(f"Validation images: {len(valid_df)}")
    print(f"Testing images: {len(ts_df)}")

    tr_gen, valid_gen, ts_gen = create_generators(
        tr_df,
        valid_df,
        ts_df,
        batch_size=batch_size,
        img_size=img_size,
    )

    class_dict = tr_gen.class_indices
    save_class_indices(class_dict, os.path.join(args.output_dir, "class_indices.json"))

    model = build_model(
        img_shape=(img_size[0], img_size[1], 3),
        num_classes=len(class_dict),
    )
    model.summary()

    hist = model.fit(
        tr_gen,
        epochs=epochs,
        validation_data=valid_gen,
        shuffle=False,
    )
    save_history(hist, os.path.join(args.output_dir, "training_history.json"))

    evaluate_model(model, tr_gen, valid_gen, ts_gen)

    if not args.no_plots:
        plot_training_history(hist)
        plot_confusion_matrix(model, ts_gen, class_dict)

    model_path = os.path.join(args.output_dir, model_name)
    model.save(model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
