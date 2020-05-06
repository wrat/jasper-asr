import numpy as np
import wave
import io
import json
from pathlib import Path
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


def asr_data_writer(output_dir, dataset_name, asr_data_source):
    dataset_dir = output_dir / Path(dataset_name)
    (dataset_dir / Path("wav")).mkdir(parents=True, exist_ok=True)
    asr_manifest = dataset_dir / Path("manifest.json")
    with asr_manifest.open("w") as mf:
        for pnr_code, audio_dur, wav_data in asr_data_source:
            pnr_af = dataset_dir / Path("wav") / Path(pnr_code).with_suffix(".wav")
            pnr_af.write_bytes(wav_data)
            rel_pnr_path = pnr_af.relative_to(dataset_dir)
            manifest = manifest_str(
                str(rel_pnr_path), audio_dur, alnum_to_asr_tokens(pnr_code)
            )
            mf.write(manifest)


def asr_manifest_reader(data_manifest_path: Path):
    print(f'reading manifest from {data_manifest_path}')
    with data_manifest_path.open("r") as pf:
        pnr_jsonl = pf.readlines()
    pnr_data = [json.loads(v) for v in pnr_jsonl]
    for p in pnr_data:
        p['audio_path'] = data_manifest_path.parent / Path(p['audio_filepath'])
        p['chars'] = Path(p['audio_filepath']).stem
        yield p


def asr_manifest_writer(asr_manifest_path: Path, manifest_str_source):
    with asr_manifest_path.open("w") as mf:
        print(f'opening {asr_manifest_path} for writing manifest')
        for mani_dict in manifest_str_source:
            manifest = manifest_str(
                mani_dict['audio_filepath'], mani_dict['duration'], mani_dict['text']
            )
            mf.write(manifest)


def main():
    for c in random_pnr_generator():
        print(c)


if __name__ == "__main__":
    main()
