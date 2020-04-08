import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from num2words import num2words


def separate_space_convert_digit_setpath():
    with Path("/home/malar/work/asr-data-utils/asr_data/pnr_data.json").open("r") as pf:
        pnr_jsonl = pf.readlines()

    pnr_data = [json.loads(i) for i in pnr_jsonl]

    new_pnr_data = []
    for i in pnr_data:
        letters = " ".join(list(i["text"]))
        num_tokens = [num2words(c) if "0" <= c <= "9" else c for c in letters]
        i["text"] = ("".join(num_tokens)).lower()
        i["audio_filepath"] = i["audio_filepath"].replace(
            "pnr_data/", "/dataset/asr_data/pnr_data/wav/"
        )
        new_pnr_data.append(i)

    new_pnr_jsonl = [json.dumps(i) for i in new_pnr_data]

    with Path("/dataset/asr_data/pnr_data/pnr_data.json").open("w") as pf:
        new_pnr_data = "\n".join(new_pnr_jsonl)  # + "\n"
        pf.write(new_pnr_data)


separate_space_convert_digit_setpath()


def split_data():
    with Path("/dataset/asr_data/pnr_data/pnr_data.json").open("r") as pf:
        pnr_jsonl = pf.readlines()
    train_pnr, test_pnr = train_test_split(pnr_jsonl, test_size=0.1)
    with Path("/dataset/asr_data/pnr_data/train_manifest.json").open("w") as pf:
        pnr_data = "".join(train_pnr)
        pf.write(pnr_data)
    with Path("/dataset/asr_data/pnr_data/test_manifest.json").open("w") as pf:
        pnr_data = "".join(test_pnr)
        pf.write(pnr_data)


split_data()


def augment_an4():
    an4_train = Path("/dataset/asr_data/an4/train_manifest.json").read_bytes()
    an4_test = Path("/dataset/asr_data/an4/test_manifest.json").read_bytes()
    pnr_train = Path("/dataset/asr_data/pnr_data/train_manifest.json").read_bytes()
    pnr_test = Path("/dataset/asr_data/pnr_data/test_manifest.json").read_bytes()

    with Path("/dataset/asr_data/an4_pnr/train_manifest.json").open("wb") as pf:
        pf.write(an4_train + pnr_train)
    with Path("/dataset/asr_data/an4_pnr/test_manifest.json").open("wb") as pf:
        pf.write(an4_test + pnr_test)


augment_an4()


def validate_data(data_file):
    with Path(data_file).open("r") as pf:
        pnr_jsonl = pf.readlines()
    for (i, s) in enumerate(pnr_jsonl):
        try:
            json.loads(s)
        except BaseException as e:
            print(f"failed on {i}")


validate_data("/dataset/asr_data/an4_pnr/test_manifest.json")
validate_data("/dataset/asr_data/an4_pnr/train_manifest.json")


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
