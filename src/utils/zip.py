import os
import zipfile


def extract_zip(zip_path):
    if not os.path.isfile(zip_path):
        print(f"Zip file does not exist or is not a file: {zip_path}")
        return

    if not zipfile.is_zipfile(zip_path):
        print(f"Not a valid zip file: {zip_path}")
        return

    extract_dir = os.path.splitext(zip_path)[0]  # Remove .zip from path

    os.makedirs(extract_dir, exist_ok=True)

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"Extracted to: {extract_dir}")
    except Exception as e:
        print(f"Failed to extract zip file: {e}")
