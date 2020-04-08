# import io
# import sys
# import json
import argparse
import logging
from pathlib import Path
from .utils import random_pnr_generator, manifest_str
from .tts.googletts import GoogleTTS
from tqdm import tqdm
import random

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_asr_data(output_dir, count):
    google_voices = GoogleTTS.voice_list()
    gtts = GoogleTTS()
    wav_dir = output_dir / Path("pnr_data")
    wav_dir.mkdir(parents=True, exist_ok=True)
    asr_manifest = output_dir / Path("pnr_data").with_suffix(".json")
    with asr_manifest.open("w") as mf:
        for pnr_code in tqdm(random_pnr_generator(count)):
            tts_code = (
                f'<speak><say-as interpret-as="verbatim">{pnr_code}</say-as></speak>'
            )
            param = random.choice(google_voices)
            param["sample_rate"] = 24000
            param["num_channels"] = 1
            wav_data = gtts.text_to_speech(text=tts_code, params=param)
            audio_dur = len(wav_data[44:]) / (2 * 24000)
            pnr_af = wav_dir / Path(pnr_code).with_suffix(".wav")
            pnr_af.write_bytes(wav_data)
            rel_pnr_path = pnr_af.relative_to(output_dir)
            manifest = manifest_str(str(rel_pnr_path), audio_dur, pnr_code)
            mf.write(manifest)


def arg_parser():
    prog = Path(__file__).stem
    parser = argparse.ArgumentParser(
        prog=prog, description=f"generates asr training data"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./train/asr_data"),
        help="directory to output asr data",
    )
    parser.add_argument(
        "--count", type=int, default=3, help="number of datapoints to generate"
    )
    return parser


def main():
    parser = arg_parser()
    args = parser.parse_args()
    generate_asr_data(**vars(args))


if __name__ == "__main__":
    main()
