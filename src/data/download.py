from pathlib import Path
import subprocess

from src.paths import DATA_RAW_DIR
from src.config import DATASET_NAME


# Dataset Kaggle (identificador oficial)
KAGGLE_DATASET_REF = "abdelazizel7or/airline-delay-cause"


def download_dataset(force: bool = False) -> None:
    """
    Baixa o dataset do Kaggle e salva em data/raw/.

    Parâmetros
    ----------
    force : bool
        Se True, força o download mesmo que os arquivos já existam.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_RAW_DIR / "dataset.zip"

    if zip_path.exists() and not force:
        print("Dataset já existe. Use force=True para baixar novamente.")
        return

    print(f"Baixando dataset: {DATASET_NAME}")

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        KAGGLE_DATASET_REF,
        "-p",
        str(DATA_RAW_DIR),
        "--force" if force else "",
    ]

    # Remove strings vazias
    command = [c for c in command if c]

    subprocess.run(command, check=True)

    print("Download concluído.")


if __name__ == "__main__":
    download_dataset()
    