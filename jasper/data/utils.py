import io
import os
import json
import wave
from pathlib import Path
from itertools import product
from functools import partial
from math import floor
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pymongo
from slugify import slugify
from num2words import num2words
from jasper.client import transcribe_gen
from nemo.collections.asr.metrics import word_error_rate
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tqdm import tqdm


def manifest_str(path, dur, text):
    return (
        json.dumps({"audio_filepath": path, "duration": round(dur, 1), "text": text})
        + "\n"
    )


def wav_bytes(audio_bytes, frame_rate=24000):
    wf_b = io.BytesIO()
    with wave.open(wf_b, mode="w") as wf:
        wf.setnchannels(1)
        wf.setframerate(frame_rate)
        wf.setsampwidth(2)
        wf.writeframesraw(audio_bytes)
    return wf_b.getvalue()


def random_pnr_generator(count=10000):
    LENGTH = 3

    # alphabet = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    numeric = list("0123456789")
    np_alphabet = np.array(alphabet, dtype="|S1")
    np_numeric = np.array(numeric, dtype="|S1")
    np_alpha_codes = np.random.choice(np_alphabet, [count, LENGTH])
    np_num_codes = np.random.choice(np_numeric, [count, LENGTH])
    np_code_seed = np.concatenate((np_alpha_codes, np_num_codes), axis=1).T
    np.random.shuffle(np_code_seed)
    np_codes = np_code_seed.T
    codes = [(b"".join(np_codes[i])).decode("utf-8") for i in range(len(np_codes))]
    return codes


def alnum_to_asr_tokens(text):
    letters = " ".join(list(text))
    num_tokens = [num2words(c) if "0" <= c <= "9" else c for c in letters]
    return ("".join(num_tokens)).lower()


def tscript_uuid_fname(transcript):
    return str(uuid4()) + "_" + slugify(transcript, max_length=8)


def asr_data_writer(output_dir, dataset_name, asr_data_source, verbose=False):
    dataset_dir = output_dir / Path(dataset_name)
    (dataset_dir / Path("wav")).mkdir(parents=True, exist_ok=True)
    asr_manifest = dataset_dir / Path("manifest.json")
    num_datapoints = 0
    with asr_manifest.open("w") as mf:
        print(f"writing manifest to {asr_manifest}")
        for transcript, audio_dur, wav_data in asr_data_source:
            fname = tscript_uuid_fname(transcript)
            audio_file = dataset_dir / Path("wav") / Path(fname).with_suffix(".wav")
            audio_file.write_bytes(wav_data)
            rel_pnr_path = audio_file.relative_to(dataset_dir)
            manifest = manifest_str(str(rel_pnr_path), audio_dur, transcript)
            mf.write(manifest)
            if verbose:
                print(f"writing '{transcript}' of duration {audio_dur}")
            num_datapoints += 1
    return num_datapoints


def ui_dump_manifest_writer(output_dir, dataset_name, asr_data_source, verbose=False):
    dataset_dir = output_dir / Path(dataset_name)
    (dataset_dir / Path("wav")).mkdir(parents=True, exist_ok=True)
    ui_dump_file = dataset_dir / Path("ui_dump.json")
    (dataset_dir / Path("wav_plots")).mkdir(parents=True, exist_ok=True)
    asr_manifest = dataset_dir / Path("manifest.json")
    num_datapoints = 0
    ui_dump = {
        "use_domain_asr": False,
        "annotation_only": False,
        "enable_plots": True,
        "data": [],
    }
    data_funcs = []
    transcriber_pretrained = transcribe_gen(asr_port=8044)
    with asr_manifest.open("w") as mf:
        print(f"writing manifest to {asr_manifest}")

        def data_fn(
            transcript,
            audio_dur,
            wav_data,
            caller_name,
            aud_seg,
            fname,
            audio_path,
            num_datapoints,
            rel_pnr_path,
        ):
            pretrained_result = transcriber_pretrained(aud_seg.raw_data)
            pretrained_wer = word_error_rate([transcript], [pretrained_result])
            wav_plot_path = (
                dataset_dir / Path("wav_plots") / Path(fname).with_suffix(".png")
            )
            if not wav_plot_path.exists():
                plot_seg(wav_plot_path, audio_path)
            return {
                "audio_filepath": str(rel_pnr_path),
                "duration": round(audio_dur, 1),
                "text": transcript,
                "real_idx": num_datapoints,
                "audio_path": audio_path,
                "spoken": transcript,
                "caller": caller_name,
                "utterance_id": fname,
                "pretrained_asr": pretrained_result,
                "pretrained_wer": pretrained_wer,
                "plot_path": str(wav_plot_path),
            }

        for transcript, audio_dur, wav_data, caller_name, aud_seg in asr_data_source:
            fname = str(uuid4()) + "_" + slugify(transcript, max_length=8)
            audio_file = dataset_dir / Path("wav") / Path(fname).with_suffix(".wav")
            audio_file.write_bytes(wav_data)
            audio_path = str(audio_file)
            rel_pnr_path = audio_file.relative_to(dataset_dir)
            manifest = manifest_str(str(rel_pnr_path), audio_dur, transcript)
            mf.write(manifest)
            data_funcs.append(
                partial(
                    data_fn,
                    transcript,
                    audio_dur,
                    wav_data,
                    caller_name,
                    aud_seg,
                    fname,
                    audio_path,
                    num_datapoints,
                    rel_pnr_path,
                )
            )
            num_datapoints += 1
    with ThreadPoolExecutor() as exe:
        print("starting all plot/transcription tasks")
        dump_data = list(
            tqdm(
                exe.map(lambda x: x(), data_funcs),
                position=0,
                leave=True,
                total=len(data_funcs),
            )
        )
    ui_dump["data"] = dump_data
    ExtendedPath(ui_dump_file).write_json(ui_dump)
    return num_datapoints


def asr_manifest_reader(data_manifest_path: Path):
    print(f"reading manifest from {data_manifest_path}")
    with data_manifest_path.open("r") as pf:
        pnr_jsonl = pf.readlines()
    pnr_data = [json.loads(v) for v in pnr_jsonl]
    for p in pnr_data:
        p["audio_path"] = data_manifest_path.parent / Path(p["audio_filepath"])
        p["text"] = p["text"].strip()
        yield p


def asr_manifest_writer(asr_manifest_path: Path, manifest_str_source):
    with asr_manifest_path.open("w") as mf:
        print(f"opening {asr_manifest_path} for writing manifest")
        for mani_dict in manifest_str_source:
            manifest = manifest_str(
                mani_dict["audio_filepath"], mani_dict["duration"], mani_dict["text"]
            )
            mf.write(manifest)


class ExtendedPath(type(Path())):
    """docstring for ExtendedPath."""

    def read_json(self):
        print(f"reading json from {self}")
        with self.open("r") as jf:
            return json.load(jf)

    def write_json(self, data):
        print(f"writing json to {self}")
        self.parent.mkdir(parents=True, exist_ok=True)
        with self.open("w") as jf:
            return json.dump(data, jf, indent=2)


def get_mongo_coll(uri="mongodb://localhost:27017/test.calls"):
    ud = pymongo.uri_parser.parse_uri(uri)
    conn = pymongo.MongoClient(uri)
    return conn[ud["database"]][ud["collection"]]


def get_mongo_conn(host="", port=27017, db="test", col="calls"):
    mongo_host = host if host else os.environ.get("MONGO_HOST", "localhost")
    mongo_uri = f"mongodb://{mongo_host}:{port}/"
    return pymongo.MongoClient(mongo_uri)[db][col]


def strip_silence(sound):
    from pydub.silence import detect_leading_silence

    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())
    duration = len(sound)
    return sound[start_trim : duration - end_trim]


def plot_seg(wav_plot_path, audio_path):
    fig = plt.Figure()
    ax = fig.add_subplot()
    (y, sr) = librosa.load(audio_path)
    librosa.display.waveplot(y=y, sr=sr, ax=ax)
    with wav_plot_path.open("wb") as wav_plot_f:
        fig.set_tight_layout(True)
        fig.savefig(wav_plot_f, format="png", dpi=50)


def generate_dates():

    days = [i for i in range(1, 32)]
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    # ordinal from https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement

    def ordinal(n):
        return "%d%s" % (
            n,
            "tsnrhtdd"[(floor(n / 10) % 10 != 1) * (n % 10 < 4) * n % 10 :: 4],
        )

    def canon_vars(d, m):
        return [
            ordinal(d) + " " + m,
            m + " " + ordinal(d),
            ordinal(d) + " of " + m,
            m + " the " + ordinal(d),
            str(d) + " " + m,
            m + " " + str(d),
        ]

    return [dm for d, m in product(days, months) for dm in canon_vars(d, m)]


def main():
    for c in random_pnr_generator():
        print(c)


if __name__ == "__main__":
    main()
