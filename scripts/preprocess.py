import argparse
import os
import random
import shutil


def split_data(
    raw_train_dir: str,
    raw_test_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Делит данные из raw_train_dir на три сплита:
      - output_dir/splits/train
      - output_dir/splits/val
      - output_dir/splits/test

    Train/Val берутся из raw_train_dir (делим по классам),
    Test полностью копируется из raw_test_dir.
    """
    random.seed(seed)

    paths = {
        "train": os.path.join(output_dir, "splits", "train"),
        "val": os.path.join(output_dir, "splits", "val"),
        "test": os.path.join(output_dir, "splits", "test"),
    }

    for p in paths.values():
        shutil.rmtree(p, ignore_errors=True)
        os.makedirs(p, exist_ok=True)

    # Train/Val
    classes = sorted(
        [
            d
            for d in os.listdir(raw_train_dir)
            if os.path.isdir(os.path.join(raw_train_dir, d))
        ]
    )
    for cls in classes:
        cls_src = os.path.join(raw_train_dir, cls)
        images = [
            f
            for f in os.listdir(cls_src)
            if f.lower().endswith((".ppm", ".png", ".jpg", ".jpeg"))
        ]
        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_imgs = images[:n_train]
        # val_imgs = images[n_train : n_train + n_val]
        temp = images[n_train:]
        val_imgs = temp[:n_val]

        for subset, img_list in [("train", train_imgs), ("val", val_imgs)]:
            dst_dir = os.path.join(paths[subset], cls)
            os.makedirs(dst_dir, exist_ok=True)
            for img in img_list:
                shutil.copy(os.path.join(cls_src, img), os.path.join(dst_dir, img))

    # Test
    if os.path.isdir(raw_test_dir):
        classes_test = sorted(
            [
                d
                for d in os.listdir(raw_test_dir)
                if os.path.isdir(os.path.join(raw_test_dir, d))
            ]
        )
        for cls in classes_test:
            cls_src = os.path.join(raw_test_dir, cls)
            dst_dir = os.path.join(paths["test"], cls)
            os.makedirs(dst_dir, exist_ok=True)
            for img in os.listdir(cls_src):
                if img.lower().endswith((".ppm", ".png", ".jpg", ".jpeg")):
                    shutil.copy(os.path.join(cls_src, img), os.path.join(dst_dir, img))


def main():
    parser = argparse.ArgumentParser(
        description="Split GTSRB: Train→train/val, Test→test"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.normpath(os.path.join(os.getcwd(), "..", "data")),
        help="Корневая папка data (с подпапкой raw/GTSRB)",
    )
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Доля для train")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Доля для val")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting"
    )
    args = parser.parse_args()

    raw_train = os.path.join(args.data_dir, "raw", "GTSRB", "Train")
    raw_test = os.path.join(args.data_dir, "raw", "GTSRB", "Test")

    if not os.path.isdir(raw_train):
        raise FileNotFoundError(f"Train folder not found: {raw_train}")
    if not os.path.isdir(raw_test):
        print(f"Warning: Test folder not found at {raw_test}, skipping test split.")

    print("Splitting data...")
    split_data(
        raw_train_dir=raw_train,
        raw_test_dir=raw_test,
        output_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print("Done! Splits are in data/splits/{train,val,test}/")


if __name__ == "__main__":
    main()
