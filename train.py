import argparse
import os
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train brain tumor MRI classifier.")
    parser.add_argument(
        "--train-dir",
        default="/kaggle/input/brain-tumor-mri-dataset/Training",
        help="Training dataset directory.",
    )
    parser.add_argument(
        "--test-dir",
        default="/kaggle/input/brain-tumor-mri-dataset/Testing",
        help="Testing dataset directory.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, nargs=2, default=(299, 299))
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--model-name", default="brain_tumor_xception.keras")
    parser.add_argument("--no-plots", action="store_true", help="Disable plots.")
    return parser.parse_args()


def main():
    args = parse_args()

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

    ensure_dir(args.output_dir)

    tr_df, valid_df, ts_df = load_datasets(args.train_dir, args.test_dir)
    print(f"Training images: {len(tr_df)}")
    print(f"Validation images: {len(valid_df)}")
    print(f"Testing images: {len(ts_df)}")

    tr_gen, valid_gen, ts_gen = create_generators(
        tr_df,
        valid_df,
        ts_df,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
    )

    class_dict = tr_gen.class_indices
    save_class_indices(class_dict, os.path.join(args.output_dir, "class_indices.json"))

    model = build_model(
        img_shape=(args.img_size[0], args.img_size[1], 3),
        num_classes=len(class_dict),
    )
    model.summary()

    hist = model.fit(
        tr_gen,
        epochs=args.epochs,
        validation_data=valid_gen,
        shuffle=False,
    )
    save_history(hist, os.path.join(args.output_dir, "training_history.json"))

    evaluate_model(model, tr_gen, valid_gen, ts_gen)

    if not args.no_plots:
        plot_training_history(hist)
        plot_confusion_matrix(model, ts_gen, class_dict)

    model_path = os.path.join(args.output_dir, args.model_name)
    model.save(model_path)
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
