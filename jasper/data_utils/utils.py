import numpy as np
import wave
import io
import json


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


def main():
    for c in random_pnr_generator():
        print(c)


if __name__ == "__main__":
    main()
