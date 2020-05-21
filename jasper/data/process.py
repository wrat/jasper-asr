import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from .utils import asr_manifest_reader, asr_manifest_writer
from typing import List
from itertools import chain
import typer

app = typer.Typer()


@app.command()
def fixate_data(dataset_path: Path):
    manifest_path = dataset_path / Path("manifest.json")
    real_manifest_path = dataset_path / Path("abs_manifest.json")

    def fix_path():
        for i in asr_manifest_reader(manifest_path):
            i["audio_filepath"] = str(dataset_path / Path(i["audio_filepath"]))
            yield i

    asr_manifest_writer(real_manifest_path, fix_path())


@app.command()
def augment_datasets(src_dataset_paths: List[Path], dest_dataset_path: Path):
    reader_list = []
    abs_manifest_path = Path("abs_manifest.json")
    for dataset_path in src_dataset_paths:
        manifest_path = dataset_path / abs_manifest_path
        reader_list.append(asr_manifest_reader(manifest_path))
    dest_dataset_path.mkdir(parents=True, exist_ok=True)
    dest_manifest_path = dest_dataset_path / abs_manifest_path
    asr_manifest_writer(dest_manifest_path, chain(*reader_list))


@app.command()
def split_data(dataset_path: Path, test_size: float = 0.1):
    manifest_path = dataset_path / Path("abs_manifest.json")
    asr_data = list(asr_manifest_reader(manifest_path))
    train_pnr, test_pnr = train_test_split(asr_data, test_size=test_size)
    asr_manifest_writer(manifest_path.with_name("train_manifest.json"), train_pnr)
    asr_manifest_writer(manifest_path.with_name("test_manifest.json"), test_pnr)


@app.command()
def validate_data(dataset_path: Path):
    for mf_type in ["train_manifest.json", "test_manifest.json"]:
        data_file = dataset_path / Path(mf_type)
        print(f"validating {data_file}.")
        with Path(data_file).open("r") as pf:
            pnr_jsonl = pf.readlines()
        for (i, s) in enumerate(pnr_jsonl):
            try:
                d = json.loads(s)
                audio_file = data_file.parent / Path(d["audio_filepath"])
                if not audio_file.exists():
                    raise OSError(f"File {audio_file} not found")
            except BaseException as e:
                print(f'failed on {i} with "{e}"')
        print(f"no errors found. seems like a valid {mf_type}.")


def main():
    app()


if __name__ == "__main__":
    main()
