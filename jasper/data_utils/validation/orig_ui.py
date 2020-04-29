import json
from pathlib import Path
import streamlit as st

# import matplotlib.pyplot as plt
# import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment
from jasper.client import transcriber_pretrained, transcriber_speller

# from pymongo import MongoClient

st.title("ASR Speller Validation")
dataset_path: Path = Path("/dataset/asr_data/call_alphanum_v3")
manifest_path = dataset_path / Path("test_manifest.json")
# print(manifest_path)
with manifest_path.open("r") as pf:
    pnr_jsonl = pf.readlines()
    pnr_data = [json.loads(i) for i in pnr_jsonl]


def main():
    # pnr_data = MongoClient("mongodb://localhost:27017/").test.asr_pnr
    # sample_no = 0
    sample_no = (
        st.slider(
            "Sample",
            min_value=1,
            max_value=len(pnr_data),
            value=1,
            step=1,
            format=None,
            key=None,
        )
        - 1
    )
    sample = pnr_data[sample_no]
    st.write(f"Sample No: {sample_no+1} of {len(pnr_data)}")
    audio_path = Path(sample["audio_filepath"])
    # st.write(f"Audio Path:{audio_path}")
    aud_seg = AudioSegment.from_wav(audio_path)  # .set_channels(1).set_sample_width(2).set_frame_rate(24000)
    st.sidebar.text("Transcription")
    st.sidebar.text(f"Pretrained:{transcriber_pretrained(aud_seg.raw_data)}")
    st.sidebar.text(f"Speller:{transcriber_speller(aud_seg.raw_data)}")
    st.sidebar.text(f"Expected: {audio_path.stem}")
    spell_text = sample["text"]
    st.sidebar.text(f"Spelled: {spell_text}")
    st.audio(audio_path.open("rb"))
    selected = st.radio("The Audio is", ("Correct", "Incorrect", "Inaudible"))
    corrected = audio_path.stem
    if selected == "Incorrect":
        corrected = st.text_input("Actual:", value=corrected)
    # content = ''
    if sample_no > 0 and st.button("Previous"):
        sample_no -= 1
    if st.button("Next"):
        st.write(sample_no, selected, corrected)
        sample_no += 1

    (y, sr) = librosa.load(audio_path)
    librosa.display.waveplot(y=y, sr=sr)
    # arr = np.random.normal(1, 1, size=100)
    # plt.hist(arr, bins=20)
    st.sidebar.pyplot()


# def main():
#     app()


if __name__ == "__main__":
    main()
