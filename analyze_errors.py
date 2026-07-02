import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Export test-set prediction errors.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--test-dir", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--class-indices", default=None)
    parser.add_argument("--output", default="reports/prediction_errors.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    import numpy as np
    from tensorflow.keras.models import load_model

    from config import get_nested, load_config
    from dataset import create_generators, load_datasets
    from utils import load_class_indices, save_prediction_errors

    cfg = load_config(args.config)
    train_dir = get_nested(cfg, "data.train_dir")
    test_dir = args.test_dir or get_nested(cfg, "data.test_dir")
    img_size = tuple(get_nested(cfg, "training.img_size", [299, 299]))
    output_dir = get_nested(cfg, "output.dir", "artifacts")
    model_name = get_nested(cfg, "output.model_name", "brain_tumor_xception.keras")
    model_path = args.model or os.path.join(output_dir, model_name)
    class_indices_path = args.class_indices or os.path.join(
        output_dir, "class_indices.json"
    )

    _, valid_df, ts_df = load_datasets(train_dir, test_dir)
    _, _, ts_gen = create_generators(valid_df, valid_df, ts_df, img_size=img_size)
    model = load_model(model_path)
    class_dict = load_class_indices(class_indices_path)
    y_pred = np.argmax(model.predict(ts_gen), axis=1)
    rows = save_prediction_errors(ts_gen, y_pred, class_dict, args.output)
    print(f"Saved {len(rows)} prediction errors to {args.output}")


if __name__ == "__main__":
    main()
