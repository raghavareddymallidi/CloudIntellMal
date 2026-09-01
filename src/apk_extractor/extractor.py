import subprocess
from pathlib import Path


def extract_apk(apk_path, output_directory):
    """
    Extract an APK into Smali files using Apktool.
    """

    apk_path = Path(apk_path)
    output_directory = Path(output_directory)

    if not apk_path.exists():
        raise FileNotFoundError(f"APK not found: {apk_path}")

    if apk_path.suffix.lower() != ".apk":
        raise ValueError("Input file must be an APK")

    output_directory.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "apktool",
        "d",
        str(apk_path),
        "-o",
        str(output_directory),
        "-f"
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True
        )

    except subprocess.CalledProcessError as error:
        print("APK extraction failed.")
        print(error.stderr)
        raise

    print(f"APK extracted successfully to: {output_directory}")

    return output_directory
