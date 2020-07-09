import io
import os
import json
import wave
from pathlib import Path
from itertools import product
from functools import partial
from math import floor
from uuid import uuid4
from urllib.parse import urlsplit
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
    transcriber_gcp = gcp_transcribe_gen()
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
            gcp_seg = aud_seg.set_frame_rate(16000)
            gcp_result = transcriber_gcp(gcp_seg.raw_data)
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
                "gcp_asr": gcp_result,
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
    dump_data = parallel_apply(lambda x: x(), data_funcs)
    # dump_data = [x() for x in tqdm(data_funcs)]
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


def asr_test_writer(out_file_path: Path, source):
    def dd_str(dd, idx):
        path = dd["audio_filepath"]
        # dur = dd["duration"]
        # return f"SAY {idx}\nPAUSE 3\nPLAY {path}\nPAUSE 3\n\n"
        return f"PAUSE 2\nPLAY {path}\nPAUSE 60\n\n"

    res_file = out_file_path.with_suffix(".result.json")
    with out_file_path.open("w") as of:
        print(f"opening {out_file_path} for writing test")
        results = []
        idx = 0
        for ui_dd in source:
            results.append(ui_dd)
            out_str = dd_str(ui_dd, idx)
            of.write(out_str)
            idx += 1
        of.write("DO_HANGUP\n")
        ExtendedPath(res_file).write_json(results)


def batch(iterable, n=1):
    ls = len(iterable)
    return [iterable[ndx : min(ndx + n, ls)] for ndx in range(0, ls, n)]


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


def get_call_logs(call_obj, s3, call_meta_dir):
    meta_s3_uri = call_obj["DataURI"]
    s3_event_url_p = urlsplit(meta_s3_uri)
    saved_meta_path = call_meta_dir / Path(Path(s3_event_url_p.path).name)
    if not saved_meta_path.exists():
        print(f"downloading : {saved_meta_path} from {meta_s3_uri}")
        s3.download_file(
            s3_event_url_p.netloc, s3_event_url_p.path[1:], str(saved_meta_path)
        )
    call_metas = json.load(saved_meta_path.open())
    return call_metas


def gcp_transcribe_gen():
    from google.cloud import speech_v1
    from google.cloud.speech_v1 import enums

    # import io
    client = speech_v1.SpeechClient()
    # local_file_path = 'resources/brooklyn_bridge.raw'

    # The language of the supplied audio
    language_code = "en-US"
    model = "phone_call"

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = 16000

    # Encoding of audio data sent. This sample sets this explicitly.
    # This field is optional for FLAC and WAV audio formats.
    encoding = enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "language_code": language_code,
        "sample_rate_hertz": sample_rate_hertz,
        "encoding": encoding,
        "model": model,
        "enable_automatic_punctuation": True,
        "max_alternatives": 10,
        "enable_word_time_offsets": True,  # used to detect start and end time of utterances
        "speech_contexts": [
            {
                "phrases": [
                    "$OOV_CLASS_ALPHANUMERIC_SEQUENCE",
                    "$OOV_CLASS_DIGIT_SEQUENCE",
                    "$TIME",
                    "$YEAR",
                ]
            },
            {
                "phrases": [
                    "A",
                    "B",
                    "C",
                    "D",
                    "E",
                    "F",
                    "G",
                    "H",
                    "I",
                    "J",
                    "K",
                    "L",
                    "M",
                    "N",
                    "O",
                    "P",
                    "Q",
                    "R",
                    "S",
                    "T",
                    "U",
                    "V",
                    "W",
                    "X",
                    "Y",
                    "Z",
                ]
            },
            {
                "phrases": [
                    "PNR is $OOV_CLASS_ALPHANUMERIC_SEQUENCE",
                    "my PNR is $OOV_CLASS_ALPHANUMERIC_SEQUENCE",
                    "my PNR number is $OOV_CLASS_ALPHANUMERIC_SEQUENCE",
                    "PNR number is $OOV_CLASS_ALPHANUMERIC_SEQUENCE",
                    "It's $OOV_CLASS_ALPHANUMERIC_SEQUENCE",
                    "$OOV_CLASS_ALPHANUMERIC_SEQUENCE is my PNR",
                ]
            },
            {"phrases": ["my name is"]},
            {"phrases": ["Number $ORDINAL", "Numeral $ORDINAL"]},
            {
                "phrases": [
                    "John Smith",
                    "Carina Hu",
                    "Travis Lim",
                    "Marvin Tan",
                    "Samuel Tan",
                    "Dawn Mathew",
                    "Dawn",
                    "Mathew",
                ]
            },
            {
                "phrases": [
                    "Beijing",
                    "Tokyo",
                    "London",
                    "19 August",
                    "7 October",
                    "11 December",
                    "17 September",
                    "19th August",
                    "7th October",
                    "11th December",
                    "17th September",
                    "ABC123",
                    "KWXUNP",
                    "XLU5K1",
                    "WL2JV6",
                    "KBS651",
                ]
            },
            {
                "phrases": [
                    "first flight",
                    "second flight",
                    "third flight",
                    "first option",
                    "second option",
                    "third option",
                    "first one",
                    "second one",
                    "third one",
                ]
            },
        ],
        "metadata": {
            "industry_naics_code_of_audio": 481111,
            "interaction_type": enums.RecognitionMetadata.InteractionType.PHONE_CALL,
        },
    }

    def sample_recognize(content):
        """
        Transcribe a short audio file using synchronous speech recognition

        Args:
          local_file_path Path to local audio file, e.g. /path/audio.wav
        """

        # with io.open(local_file_path, "rb") as f:
        #     content = f.read()
        audio = {"content": content}

        response = client.recognize(config, audio)
        for result in response.results:
            # First alternative is the most probable result
            return "/".join([alt.transcript for alt in result.alternatives])
            # print(u"Transcript: {}".format(alternative.transcript))
        return ""

    return sample_recognize


def parallel_apply(fn, iterable, workers=8):
    with ThreadPoolExecutor(max_workers=workers) as exe:
        print(f"parallelly applying {fn}")
        return [
            res
            for res in tqdm(
                exe.map(fn, iterable), position=0, leave=True, total=len(iterable)
            )
        ]


def main():
    for c in random_pnr_generator():
        print(c)


if __name__ == "__main__":
    main()
