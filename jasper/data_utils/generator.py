# import io
# import sys
# import json
import argparse
import logging
from pathlib import Path
from .utils import random_pnr_generator, asr_data_writer
from .tts.googletts import GoogleTTS
from tqdm import tqdm
import random

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def pnr_tts_streamer(count):
    google_voices = GoogleTTS.voice_list()
    gtts = GoogleTTS()
    for pnr_code in tqdm(random_pnr_generator(count)):
        tts_code = f'<speak><say-as interpret-as="verbatim">{pnr_code}</say-as></speak>'
        param = random.choice(google_voices)
        param["sample_rate"] = 24000
        param["num_channels"] = 1
        wav_data = gtts.text_to_speech(text=tts_code, params=param)
        audio_dur = len(wav_data[44:]) / (2 * 24000)
        yield pnr_code, audio_dur, wav_data


def generate_asr_data_fromtts(output_dir, dataset_name, count):
    asr_data_writer(output_dir, dataset_name, pnr_tts_streamer(count))


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
    parser.add_argument(
        "--dataset_name", type=str, default="pnr_data", help="name of the dataset"
    )
    return parser


def main():
    parser = arg_parser()
    args = parser.parse_args()
    generate_asr_data_fromtts(**vars(args))


if __name__ == "__main__":
    main()
