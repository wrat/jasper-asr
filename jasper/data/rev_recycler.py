import typer
from itertools import chain
from io import BytesIO
from pathlib import Path
import re

app = typer.Typer()


@app.command()
def extract_data(
    call_audio_dir: Path = typer.Option(Path("/dataset/rev/wavs"), show_default=True),
    call_meta_dir: Path = typer.Option(Path("/dataset/rev/jsons"), show_default=True),
    output_dir: Path = typer.Option(Path("./data"), show_default=True),
    dataset_name: str = typer.Option("rev_transribed", show_default=True),
    verbose: bool = False,
):
    from pydub import AudioSegment
    from .utils import ExtendedPath, asr_data_writer, strip_silence
    from lenses import lens
    import datetime

    call_asr_data: Path = output_dir / Path("asr_data")
    call_asr_data.mkdir(exist_ok=True, parents=True)

    def wav_event_generator(call_audio_dir):
        for wav_path in call_audio_dir.glob("**/*.wav"):
            if verbose:
                typer.echo(f"loading events for file {wav_path}")
            call_wav = AudioSegment.from_file_using_temporary_files(wav_path)
            rel_meta_path = wav_path.with_suffix(".json").relative_to(call_audio_dir)
            meta_path = call_meta_dir / rel_meta_path
            if meta_path.exists():
                events = ExtendedPath(meta_path).read_json()
                yield call_wav, wav_path, events
            else:
                typer.echo(f"missing json corresponding to {wav_path}")

    def contains_asr(x):
        return "AsrResult" in x

    def channel(n):
        def filter_func(ev):
            return (
                ev["AsrResult"]["Channel"] == n
                if "Channel" in ev["AsrResult"]
                else n == 0
            )

        return filter_func

    # def compute_endtime(call_wav, state):
    #     for (i, st) in enumerate(state):
    #         start_time = st["AsrResult"]["Alternatives"][0].get("StartTime", 0)
    #         transcript = st["AsrResult"]["Alternatives"][0]["Transcript"]
    #         if i + 1 < len(state):
    #             end_time = state[i + 1]["AsrResult"]["Alternatives"][0]["StartTime"]
    #         else:
    #             end_time = call_wav.duration_seconds
    #         full_code_seg = call_wav[start_time * 1000 : end_time * 1000]
    #         code_seg = strip_silence(full_code_seg)
    #         code_fb = BytesIO()
    #         code_seg.export(code_fb, format="wav")
    #         code_wav = code_fb.getvalue()
    #         # only starting 1 min audio has reliable alignment
    #         if start_time > 60:
    #             break
    #         # only of some audio data is present yield it
    #         if code_seg.duration_seconds >= 0.5:
    #             yield transcript, code_seg.duration_seconds, code_wav

    # def generate_call_asr_data():
    #     full_asr_data = []
    #     total_duration = 0
    #     for wav, wav_path, ev in wav_event_generator(call_audio_dir):
    #         asr_data = asr_data_generator(wav, wav_path, ev)
    #         total_duration += wav.duration_seconds
    #         full_asr_data.append(asr_data)
    #     typer.echo(f"loaded {len(full_asr_data)} calls of duration {total_duration}s")
    #     n_dps = asr_data_writer(call_asr_data, dataset_name, chain(*full_asr_data))
    #     typer.echo(f"written {n_dps} data points")

    # generate_call_asr_data()

    def time_to_msecs(time_str):
        return (
            datetime.datetime.strptime(time_str, "%H:%M:%S,%f")
            - datetime.datetime(1900, 1, 1)
        ).total_seconds() * 1000

    def dual_asr_data_generator(wav_seg, wav_path, meta):
        left_audio, right_audio = wav_seg.split_to_mono()
        channel_map = {"Agent": right_audio, "Client": left_audio}
        monologues = lens["monologues"].Each().collect()(meta)
        for monologue in monologues:
            # print(monologue["speaker_name"])
            speaker_channel = channel_map.get(monologue["speaker_name"])
            if not speaker_channel:
                print(f'unknown speaker tag {monologue["speaker_name"]} in wav:{wav_path} skipping.')
                continue
            try:
                start_time = (
                    lens["elements"]
                    .Each()
                    .Filter(lambda x: "timestamp" in x)["timestamp"]
                    .collect()(monologue)[0]
                )
                end_time = (
                    lens["elements"]
                    .Each()
                    .Filter(lambda x: "end_timestamp" in x)["end_timestamp"]
                    .collect()(monologue)[-1]
                )
            except IndexError:
                print(f'error when loading timestamp events in wav:{wav_path} skipping.')
                continue

            # offset by 500 msec to include first vad? discarded audio
            full_tscript_wav_seg = speaker_channel[time_to_msecs(start_time) - 500 : time_to_msecs(end_time)]
            tscript_wav_seg = strip_silence(full_tscript_wav_seg)
            tscript_wav_fb = BytesIO()
            tscript_wav_seg.export(tscript_wav_fb, format="wav")
            tscript_wav = tscript_wav_fb.getvalue()
            text = "".join(lens["elements"].Each()["value"].collect()(monologue))
            text_clean = re.sub(r"\[.*\]", "", text)
            yield text_clean, tscript_wav_seg.duration_seconds, tscript_wav

    def mono_asr_data_generator(wav_seg, wav_path, meta):
        monologues = lens["monologues"].Each().collect()(meta)
        for monologue in monologues:
            try:
                start_time = (
                    lens["elements"]
                    .Each()
                    .Filter(lambda x: "timestamp" in x)["timestamp"]
                    .collect()(monologue)[0]
                )
                end_time = (
                    lens["elements"]
                    .Each()
                    .Filter(lambda x: "end_timestamp" in x)["end_timestamp"]
                    .collect()(monologue)[-1]
                )
            except IndexError:
                print(f'error when loading timestamp events in wav:{wav_path} skipping.')
                continue

            # offset by 500 msec to include first vad? discarded audio
            full_tscript_wav_seg = wav_seg[time_to_msecs(start_time) - 500 : time_to_msecs(end_time)]
            tscript_wav_seg = strip_silence(full_tscript_wav_seg)
            tscript_wav_fb = BytesIO()
            tscript_wav_seg.export(tscript_wav_fb, format="wav")
            tscript_wav = tscript_wav_fb.getvalue()
            text = "".join(lens["elements"].Each()["value"].collect()(monologue))
            text_clean = re.sub(r"\[.*\]", "", text)
            yield text_clean, tscript_wav_seg.duration_seconds, tscript_wav

    def generate_rev_asr_data():
        full_asr_data = []
        total_duration = 0
        for wav, wav_path, ev in wav_event_generator(call_audio_dir):
            if wav.channels > 2:
                print(f'skipping many channel audio {wav_path}')
            asr_data_generator = mono_asr_data_generator if wav.channels == 1 else dual_asr_data_generator
            asr_data = asr_data_generator(wav, wav_path, ev)
            total_duration += wav.duration_seconds
            full_asr_data.append(asr_data)
        typer.echo(f"loaded {len(full_asr_data)} calls of duration {total_duration}s")
        n_dps = asr_data_writer(call_asr_data, dataset_name, chain(*full_asr_data))
        typer.echo(f"written {n_dps} data points")

    generate_rev_asr_data()
    # DEBUG
    # data = list(wav_event_generator(call_audio_dir))
    # wav_seg, wav_path, meta = data[0]
    # left_audio, right_audio = wav_seg.split_to_mono()
    # channel_map = {"Agent": right_audio, "Client": left_audio}
    # # data[0][2]['speakers']
    # # data[0][1]
    # monologues = lens["monologues"].Each().collect()(meta)
    # for monologue in monologues:
    #     # print(monologue["speaker_name"])
    #     speaker_channel = channel_map.get(monologue["speaker_name"])
    #     # monologue = monologues[0]
    #     # monologue
    #     start_time = (
    #         lens["elements"]
    #         .Each()
    #         .Filter(lambda x: "timestamp" in x)["timestamp"]
    #         .collect()(monologue)[0]
    #     )
    #     end_time = (
    #         lens["elements"]
    #         .Each()
    #         .Filter(lambda x: "end_timestamp" in x)["end_timestamp"]
    #         .collect()(monologue)[-1]
    #     )
    #     start_time, end_time
    #
    #     # offset by 500 msec to include first vad? discarded audio
    #     speaker_channel[time_to_msecs(start_time) - 500 : time_to_msecs(end_time)]
    #
    #     # start_time = lens["elements"][0].get()(monologue)['timestamp']
    #     # end_time = lens["elements"][-1].get()(monologue)['timestamp']
    #     text = "".join(lens["elements"].Each()["value"].collect()(monologue))
    #     text_clean = re.sub(r"\[.*\]", "", text)
    #     # print(text)
    #     # print(text_clean)


def main():
    app()


if __name__ == "__main__":
    main()
