import os
import shutil

import kagglehub


def download_data():
    """
    Скачивает GTSRB (Train + Test) через kagglehub
    и распаковывает в data/raw/GTSRB/Train и data/raw/GTSRB/Test
    """

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_raw_dir = os.path.join(project_root, "data", "raw", "GTSRB")
    os.makedirs(data_raw_dir, exist_ok=True)

    train_folder = os.path.join(data_raw_dir, "Train")
    test_folder = os.path.join(data_raw_dir, "Test")

    if os.path.isdir(train_folder) and len(os.listdir(train_folder)) > 0:
        print(
            f"[dvc_utils] 'Train' data already exists at {train_folder},\
            skipping download."
        )

        if os.path.isdir(test_folder) and len(os.listdir(test_folder)) > 0:
            print(f"[dvc_utils] 'Test' data also exists at {test_folder}.")
            return
        else:
            pass

    print("[dvc_utils] Downloading GTSRB dataset (Train + Test) via kagglehub...")
    out_path = kagglehub.dataset_download(
        "meowmeowmeowmeowmeow/gtsrb-german-traffic-sign"
    )
    print(f"[dvc_utils] Raw dataset extracted to {out_path}")

    extracted_train = os.path.join(out_path, "Train")
    extracted_test = os.path.join(out_path, "Test")
    if not os.path.isdir(extracted_train):
        raise FileNotFoundError(
            f"[dvc_utils] Expected folder '{extracted_train}'\
            not found after extraction."
        )
    if not os.path.isdir(extracted_test):
        print(
            f"[dvc_utils] Warning: folder '{extracted_test}' \
            not found. Skipping Test."
        )
    else:
        print(f"[dvc_utils] Found 'Test' folder at {extracted_test}")

    for folder_name in ("Train", "Test"):
        src_folder = os.path.join(out_path, folder_name)
        dst_folder = os.path.join(data_raw_dir, folder_name)
        if os.path.exists(dst_folder):
            shutil.rmtree(dst_folder)
        print(f"[dvc_utils] Moving '{src_folder}' → '{dst_folder}'")
        shutil.move(src_folder, dst_folder)

    shutil.rmtree(out_path)
    print(f"[dvc_utils] Cleanup: removed temporary '{out_path}'.")
    print(
        f"[dvc_utils] Download and move completed. Now you have:\n"
        f"    {train_folder}/\n"
        f"    {test_folder}/\n"
        f"Each contains subfolders-классы с изображениями."
    )
