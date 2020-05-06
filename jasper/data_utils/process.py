import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from .utils import alnum_to_asr_tokens, asr_manifest_reader, asr_manifest_writer
import typer

app = typer.Typer()


@app.command()
def separate_space_convert_digit_setpath():
    with Path("/home/malar/work/asr-data-utils/asr_data/pnr_data.json").open("r") as pf:
        pnr_jsonl = pf.readlines()

    pnr_data = [json.loads(i) for i in pnr_jsonl]

    new_pnr_data = []
    for i in pnr_data:
        i["text"] = alnum_to_asr_tokens(i["text"])
        i["audio_filepath"] = i["audio_filepath"].replace(
            "pnr_data/", "/dataset/asr_data/pnr_data/wav/"
        )
        new_pnr_data.append(i)

    new_pnr_jsonl = [json.dumps(i) for i in new_pnr_data]

    with Path("/dataset/asr_data/pnr_data/pnr_data.json").open("w") as pf:
        new_pnr_data = "\n".join(new_pnr_jsonl)  # + "\n"
        pf.write(new_pnr_data)


@app.command()
def split_data(dataset_path: Path, test_size: float = 0.1):
    manifest_path = dataset_path / Path("abs_manifest.json")
    asr_data = list(asr_manifest_reader(manifest_path))
    train_pnr, test_pnr = train_test_split(asr_data, test_size=test_size)
    asr_manifest_writer(manifest_path.with_name("train_manifest.json"), train_pnr)
    asr_manifest_writer(manifest_path.with_name("test_manifest.json"), test_pnr)


@app.command()
def fixate_data(dataset_path: Path):
    manifest_path = dataset_path / Path("manifest.json")
    real_manifest_path = dataset_path / Path("abs_manifest.json")

    def fix_path():
        for i in asr_manifest_reader(manifest_path):
            i["audio_filepath"] = str(dataset_path / Path(i["audio_filepath"]))
            yield i

    asr_manifest_writer(real_manifest_path, fix_path())

    # with manifest_path.open("r") as pf:
    #     pnr_jsonl = pf.readlines()
    #     pnr_data = [json.loads(i) for i in pnr_jsonl]
    #     new_pnr_data = []
    #     for i in pnr_data:
    #         i["audio_filepath"] = str(dataset_path / Path(i["audio_filepath"]))
    #         new_pnr_data.append(i)
    #     new_pnr_jsonl = [json.dumps(i) for i in new_pnr_data]
    #     real_manifest_path = dataset_path / Path("abs_manifest.json")
    #     with real_manifest_path.open("w") as pf:
    #         new_pnr_data = "\n".join(new_pnr_jsonl)  # + "\n"
    #         pf.write(new_pnr_data)


@app.command()
def augment_an4():
    an4_train = Path("/dataset/asr_data/an4/train_manifest.json").read_bytes()
    an4_test = Path("/dataset/asr_data/an4/test_manifest.json").read_bytes()
    pnr_train = Path("/dataset/asr_data/pnr_data/train_manifest.json").read_bytes()
    pnr_test = Path("/dataset/asr_data/pnr_data/test_manifest.json").read_bytes()

    with Path("/dataset/asr_data/an4_pnr/train_manifest.json").open("wb") as pf:
        pf.write(an4_train + pnr_train)
    with Path("/dataset/asr_data/an4_pnr/test_manifest.json").open("wb") as pf:
        pf.write(an4_test + pnr_test)


# augment_an4()


@app.command()
def validate_data(data_file: Path):
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
    print("no errors found. seems like a valid manifest.")


def main():
    app()


if __name__ == "__main__":
    main()

# def convert_digits(data_file="/dataset/asr_data/an4_pnr/test_manifest.json"):
#     with Path(data_file).open("r") as pf:
#         pnr_jsonl = pf.readlines()
#
#     pnr_data = [json.loads(i) for i in pnr_jsonl]
#     new_pnr_data = []
#     for i in pnr_data:
#         num_tokens = [num2words(c) for c in i["text"] if "0" <= c <= "9"]
#         i["text"] = "".join(num_tokens)
#         new_pnr_data.append(i)
#
#     new_pnr_jsonl = [json.dumps(i) for i in new_pnr_data]
#
#     with Path(data_file).open("w") as pf:
#         new_pnr_data = "\n".join(new_pnr_jsonl)  # + "\n"
#         pf.write(new_pnr_data)
#
#
# convert_digits(data_file="/dataset/asr_data/an4_pnr/train_manifest.json")
