import pymongo
import typer

# import matplotlib.pyplot as plt
from pathlib import Path
import json
import shutil

# import pandas as pd
from pydub import AudioSegment

# from .jasper_client import transcriber_pretrained, transcriber_speller
from jasper.data_utils.validation.jasper_client import (
    transcriber_pretrained,
    transcriber_speller,
)
from jasper.data_utils.utils import alnum_to_asr_tokens

# import importlib
# import jasper.data_utils.utils
# importlib.reload(jasper.data_utils.utils)
from jasper.data_utils.utils import asr_manifest_reader, asr_manifest_writer
from nemo.collections.asr.metrics import word_error_rate

# from tqdm import tqdm as tqdm_base
from tqdm import tqdm

app = typer.Typer()


@app.command()
def dump_corrections(dump_path: Path = Path("./data/corrections.json")):
    col = pymongo.MongoClient("mongodb://localhost:27017/").test.asr_validation

    cursor_obj = col.find({"type": "correction"}, projection={"_id": False})
    corrections = [c for c in cursor_obj]
    dump_f = dump_path.open("w")
    json.dump(corrections, dump_f, indent=2)
    dump_f.close()


def preprocess_datapoint(idx, rel, sample):
    res = dict(sample)
    res["real_idx"] = idx
    audio_path = rel / Path(sample["audio_filepath"])
    res["audio_path"] = str(audio_path)
    res["gold_chars"] = audio_path.stem
    res["gold_phone"] = sample["text"]
    aud_seg = (
        AudioSegment.from_wav(audio_path)
        .set_channels(1)
        .set_sample_width(2)
        .set_frame_rate(24000)
    )
    res["pretrained_asr"] = transcriber_pretrained(aud_seg.raw_data)
    res["speller_asr"] = transcriber_speller(aud_seg.raw_data)
    res["wer"] = word_error_rate([res["gold_phone"]], [res["speller_asr"]])
    return res


def load_dataset(data_manifest_path: Path):
    typer.echo(f"Using data manifest:{data_manifest_path}")
    with data_manifest_path.open("r") as pf:
        pnr_jsonl = pf.readlines()
        pnr_data = [
            preprocess_datapoint(i, data_manifest_path.parent, json.loads(v))
            for i, v in enumerate(tqdm(pnr_jsonl, position=0, leave=True))
        ]
    result = sorted(pnr_data, key=lambda x: x["wer"], reverse=True)
    return result


@app.command()
def dump_processed_data(
    data_manifest_path: Path = Path("./data/asr_data/call_alphanum/manifest.json"),
    dump_path: Path = Path("./data/processed_data.json"),
):
    typer.echo(f"Using data manifest:{data_manifest_path}")
    with data_manifest_path.open("r") as pf:
        pnr_jsonl = pf.readlines()
        pnr_data = [
            preprocess_datapoint(i, data_manifest_path.parent, json.loads(v))
            for i, v in enumerate(tqdm(pnr_jsonl, position=0, leave=True))
        ]
    result = sorted(pnr_data, key=lambda x: x["wer"], reverse=True)
    dump_path = Path("./data/processed_data.json")
    dump_f = dump_path.open("w")
    json.dump(result, dump_f, indent=2)
    dump_f.close()


@app.command()
def fill_unannotated(
    processed_data_path: Path = Path("./data/processed_data.json"),
    corrections_path: Path = Path("./data/corrections.json"),
):
    processed_data = json.load(processed_data_path.open())
    corrections = json.load(corrections_path.open())
    annotated_codes = {c["code"] for c in corrections}
    all_codes = {c["gold_chars"] for c in processed_data}
    unann_codes = all_codes - annotated_codes
    mongo_conn = pymongo.MongoClient("mongodb://localhost:27017/").test.asr_validation
    for c in unann_codes:
        mongo_conn.find_one_and_update(
            {"type": "correction", "code": c},
            {"$set": {"value": {"status": "Inaudible", "correction": ""}}},
            upsert=True,
        )


@app.command()
def update_corrections(
    data_manifest_path: Path = Path("./data/asr_data/call_alphanum/manifest.json"),
    processed_data_path: Path = Path("./data/processed_data.json"),
    corrections_path: Path = Path("./data/corrections.json"),
):
    def correct_manifest(manifest_data_gen, corrections_path):
        corrections = json.load(corrections_path.open())
        correct_set = {
            c["code"] for c in corrections if c["value"]["status"] == "Correct"
        }
        # incorrect_set = {c["code"] for c in corrections if c["value"]["status"] == "Inaudible"}
        correction_map = {
            c["code"]: c["value"]["correction"]
            for c in corrections
            if c["value"]["status"] == "Incorrect"
        }
        # for d in manifest_data_gen:
        #     if d["chars"] in incorrect_set:
        #         d["audio_path"].unlink()
        renamed_set = set()
        for d in manifest_data_gen:
            if d["chars"] in correct_set:
                yield {
                    "audio_filepath": d["audio_filepath"],
                    "duration": d["duration"],
                    "text": d["text"],
                }
            elif d["chars"] in correction_map:
                correct_text = correction_map[d["chars"]]
                renamed_set.add(correct_text)
                new_name = str(Path(correct_text).with_suffix(".wav"))
                d["audio_path"].replace(d["audio_path"].with_name(new_name))
                new_filepath = str(Path(d["audio_filepath"]).with_name(new_name))
                yield {
                    "audio_filepath": new_filepath,
                    "duration": d["duration"],
                    "text": alnum_to_asr_tokens(correct_text),
                }
            else:
                # don't delete if another correction points to an old file
                if d["chars"] not in renamed_set:
                    d["audio_path"].unlink()
                else:
                    print(f'skipping deletion of correction:{d["chars"]}')

    typer.echo(f"Using data manifest:{data_manifest_path}")
    dataset_dir = data_manifest_path.parent
    dataset_name = dataset_dir.name
    backup_dir = dataset_dir.with_name(dataset_name + ".bkp")
    if not backup_dir.exists():
        typer.echo(f"backing up to :{backup_dir}")
        shutil.copytree(str(dataset_dir), str(backup_dir))
    manifest_gen = asr_manifest_reader(data_manifest_path)
    corrected_manifest = correct_manifest(manifest_gen, corrections_path)
    new_data_manifest_path = data_manifest_path.with_name("manifest.new")
    asr_manifest_writer(new_data_manifest_path, corrected_manifest)
    new_data_manifest_path.replace(data_manifest_path)


def main():
    app()


if __name__ == "__main__":
    main()
