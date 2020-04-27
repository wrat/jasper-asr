import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from .utils import alnum_to_asr_tokens
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
def split_data(manifest_path: Path = Path("/dataset/asr_data/pnr_data/pnr_data.json")):
    with manifest_path.open("r") as pf:
        pnr_jsonl = pf.readlines()
    train_pnr, test_pnr = train_test_split(pnr_jsonl, test_size=0.1)
    with (manifest_path.parent / Path("train_manifest.json")).open("w") as pf:
        pnr_data = "".join(train_pnr)
        pf.write(pnr_data)
    with (manifest_path.parent / Path("test_manifest.json")).open("w") as pf:
        pnr_data = "".join(test_pnr)
        pf.write(pnr_data)


@app.command()
def fix_path(
    dataset_path: Path = Path("/dataset/asr_data/call_alphanum"),
):
    manifest_path = dataset_path / Path('manifest.json')
    with manifest_path.open("r") as pf:
        pnr_jsonl = pf.readlines()
        pnr_data = [json.loads(i) for i in pnr_jsonl]
        new_pnr_data = []
        for i in pnr_data:
            i["audio_filepath"] = str(dataset_path / Path(i["audio_filepath"]))
            new_pnr_data.append(i)
        new_pnr_jsonl = [json.dumps(i) for i in new_pnr_data]
        real_manifest_path = dataset_path / Path('real_manifest.json')
        with real_manifest_path.open("w") as pf:
            new_pnr_data = "\n".join(new_pnr_jsonl)  # + "\n"
            pf.write(new_pnr_data)


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
def validate_data(data_file: Path = Path("/dataset/asr_data/call_alphanum/train_manifest.json")):
    with Path(data_file).open("r") as pf:
        pnr_jsonl = pf.readlines()
    for (i, s) in enumerate(pnr_jsonl):
        try:
            json.loads(s)
        except BaseException as e:
            print(f"failed on {i}")


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
