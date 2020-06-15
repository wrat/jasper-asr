import numpy as np
import wave
import io
import os
import json
from pathlib import Path

import pymongo
from slugify import slugify
from uuid import uuid4
from num2words import num2words


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


def asr_data_writer(output_dir, dataset_name, asr_data_source, verbose=False):
    dataset_dir = output_dir / Path(dataset_name)
    (dataset_dir / Path("wav")).mkdir(parents=True, exist_ok=True)
    asr_manifest = dataset_dir / Path("manifest.json")
    num_datapoints = 0
    with asr_manifest.open("w") as mf:
        for transcript, audio_dur, wav_data in asr_data_source:
            fname = str(uuid4()) + "_" + slugify(transcript, max_length=8)
            pnr_af = dataset_dir / Path("wav") / Path(fname).with_suffix(".wav")
            pnr_af.write_bytes(wav_data)
            rel_pnr_path = pnr_af.relative_to(dataset_dir)
            manifest = manifest_str(str(rel_pnr_path), audio_dur, transcript)
            mf.write(manifest)
            if verbose:
                print(f"writing '{transcript}' of duration {audio_dur}")
            num_datapoints += 1
    return num_datapoints


def asr_manifest_reader(data_manifest_path: Path):
    print(f"reading manifest from {data_manifest_path}")
    with data_manifest_path.open("r") as pf:
        pnr_jsonl = pf.readlines()
    pnr_data = [json.loads(v) for v in pnr_jsonl]
    for p in pnr_data:
        p["audio_path"] = data_manifest_path.parent / Path(p["audio_filepath"])
        p["chars"] = Path(p["audio_filepath"]).stem
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
        print(f'reading json from {self}')
        with self.open("r") as jf:
            return json.load(jf)

    def write_json(self, data):
        print(f'writing json to {self}')
        self.parent.mkdir(parents=True, exist_ok=True)
        with self.open("w") as jf:
            return json.dump(data, jf, indent=2)


def get_mongo_coll(uri="mongodb://localhost:27017/test.calls"):
    ud = pymongo.uri_parser.parse_uri(uri)
    conn = pymongo.MongoClient(uri)
    return conn[ud['database']][ud['collection']]


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


def main():
    for c in random_pnr_generator():
        print(c)


if __name__ == "__main__":
    main()
