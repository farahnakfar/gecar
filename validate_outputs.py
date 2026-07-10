import os
import sys
from PIL import Image

def validate_outputs():
    output_dir = "output_images"
    expected_files = [
        "IMG_0026.JPG", "preview_IMG_0026.JPG",
        "IMG_0027.JPG", "preview_IMG_0027.JPG",
        "IMG_0028.JPG", "preview_IMG_0028.JPG",
        "IMG_0029.JPG", "preview_IMG_0029.JPG"
    ]

    print("Validating detection outputs...")

    errors = 0
    for filename in expected_files:
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            print(f"[-] Missing expected file: {filepath}")
            errors += 1
            continue

        file_size = os.path.getsize(filepath)
        if file_size == 0:
            print(f"[-] File is empty: {filepath}")
            errors += 1
            continue

        try:
            with Image.open(filepath) as img:
                img.verify()
            print(f"[+] Validated: {filepath} ({file_size} bytes)")
        except Exception as e:
            print(f"[-] Invalid image file {filepath}: {e}")
            errors += 1

    if errors == 0:
        print("[+] All output image validation checks passed successfully!")
        sys.exit(0)
    else:
        print(f"[-] Validation failed with {errors} error(s).")
        sys.exit(1)

if __name__ == "__main__":
    validate_outputs()
