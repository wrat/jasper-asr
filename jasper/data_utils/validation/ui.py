import json
from io import BytesIO
from pathlib import Path

import streamlit as st
from nemo.collections.asr.metrics import word_error_rate
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
from pydub import AudioSegment
import pymongo
import typer
from .jasper_client import transcriber_pretrained, transcriber_speller
from .st_rerun import rerun

app = typer.Typer()
st.title("ASR Speller Validation")


if not hasattr(st, "mongo_connected"):
    st.mongoclient = pymongo.MongoClient(
        "mongodb://localhost:27017/"
    ).test.asr_validation
    mongo_conn = st.mongoclient

    def current_cursor_fn():
        # mongo_conn = st.mongoclient
        cursor_obj = mongo_conn.find_one({"type": "current_cursor"})
        cursor_val = cursor_obj["cursor"]
        return cursor_val

    def update_cursor_fn(val=0):
        mongo_conn.find_one_and_update(
            {"type": "current_cursor"},
            {"$set": {"type": "current_cursor", "cursor": val}},
            upsert=True,
        )
        rerun()

    def get_correction_entry_fn(code):
        # mongo_conn = st.mongoclient
        # cursor_obj = mongo_conn.find_one({"type": "correction", "code": code})
        # cursor_val = cursor_obj["cursor"]
        return mongo_conn.find_one(
            {"type": "correction", "code": code}, projection={"_id": False}
        )

    def update_entry_fn(code, value):
        mongo_conn.find_one_and_update(
            {"type": "correction", "code": code},
            {"$set": {"value": value}},
            upsert=True,
        )

    cursor_obj = mongo_conn.find_one({"type": "current_cursor"})
    if not cursor_obj:
        update_cursor_fn(0)
    st.get_current_cursor = current_cursor_fn
    st.update_cursor = update_cursor_fn
    st.get_correction_entry = get_correction_entry_fn
    st.update_entry = update_entry_fn
    st.mongo_connected = True


# def clear_mongo_corrections():
#     col = pymongo.MongoClient("mongodb://localhost:27017/").test.asr_validation
#     col.delete_many({"type": "correction"})


def preprocess_datapoint(idx, rel, sample):
    res = dict(sample)
    res["real_idx"] = idx
    audio_path = rel / Path(sample["audio_filepath"])
    res["audio_path"] = audio_path
    res["gold_chars"] = audio_path.stem
    aud_seg = (
        AudioSegment.from_wav(audio_path)
        .set_channels(1)
        .set_sample_width(2)
        .set_frame_rate(24000)
    )
    res["pretrained_asr"] = transcriber_pretrained(aud_seg.raw_data)
    res["speller_asr"] = transcriber_speller(aud_seg.raw_data)
    res["wer"] = word_error_rate([res["text"]], [res["speller_asr"]])
    (y, sr) = librosa.load(audio_path)
    plt.tight_layout()
    librosa.display.waveplot(y=y, sr=sr)
    wav_plot_f = BytesIO()
    plt.savefig(wav_plot_f, format="png", dpi=50)
    plt.close()
    wav_plot_f.seek(0)
    res["plot_png"] = wav_plot_f
    return res


@st.cache(hash_funcs={"rpyc.core.netref.builtins.method": lambda _: None})
def preprocess_dataset(data_manifest_path: Path):
    typer.echo(f"Using data manifest:{data_manifest_path}")
    with data_manifest_path.open("r") as pf:
        pnr_jsonl = pf.readlines()
        pnr_data = [
            preprocess_datapoint(i, data_manifest_path.parent, json.loads(v))
            for i, v in enumerate(tqdm(pnr_jsonl))
        ]
    result = sorted(pnr_data, key=lambda x: x["wer"], reverse=True)
    return result


@app.command()
def main(manifest: Path):
    pnr_data = preprocess_dataset(manifest)
    sample_no = st.get_current_cursor()
    sample = pnr_data[sample_no]
    st.markdown(
        f"{sample_no+1} of {len(pnr_data)} : **{sample['gold_chars']}** spelled *{sample['text']}*"
    )
    new_sample = st.number_input(
        "Go To Sample:", value=sample_no + 1, min_value=1, max_value=len(pnr_data)
    )
    if new_sample != sample_no + 1:
        st.update_cursor(new_sample - 1)
    st.sidebar.title(f"Details: [{sample['real_idx']}]")
    st.sidebar.markdown(f"Gold: **{sample['gold_chars']}**")
    st.sidebar.markdown(f"Expected Speech: *{sample['text']}*")
    st.sidebar.title("Results:")
    st.sidebar.text(f"Pretrained:{sample['pretrained_asr']}")
    st.sidebar.text(f"Speller:{sample['speller_asr']}")

    st.sidebar.title(f"Speller WER: {sample['wer']:.2f}%")
    # (y, sr) = librosa.load(sample["audio_path"])
    # librosa.display.waveplot(y=y, sr=sr)
    # st.sidebar.pyplot(fig=sample["plot_fig"])
    st.sidebar.image(sample["plot_png"])
    st.audio(sample["audio_path"].open("rb"))
    corrected = sample["gold_chars"]
    correction_entry = st.get_correction_entry(sample["gold_chars"])
    selected_idx = 0
    options = ("Correct", "Incorrect", "Inaudible")
    if correction_entry:
        selected_idx = options.index(correction_entry["value"]["status"])
        corrected = correction_entry["value"]["correction"]
    selected = st.radio("The Audio is", options, index=selected_idx)
    if selected == "Incorrect":
        corrected = st.text_input("Actual:", value=corrected)
    if selected == "Inaudible":
        corrected = ""
    if st.button("Submit"):
        correct_code = corrected.replace(" ", "").upper()
        st.update_entry(
            sample["gold_chars"], {"status": selected, "correction": correct_code}
        )
        st.update_cursor(sample_no + 1)
    if correction_entry:
        st.markdown(
            f'Your Response: **{correction_entry["value"]["status"]}** Correction: **{correction_entry["value"]["correction"]}**'
        )
    # real_idx = st.text_input("Go to real-index:", value=sample['real_idx'])
    # st.markdown(
    #     ",".join(
    #         [
    #             "**" + str(p["real_idx"]) + "**"
    #             if p["real_idx"] == sample["real_idx"]
    #             else str(p["real_idx"])
    #             for p in pnr_data
    #         ]
    #     )
    # )


if __name__ == "__main__":
    try:
        app()
    except SystemExit:
        pass
