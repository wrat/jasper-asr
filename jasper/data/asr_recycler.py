import typer
from itertools import chain
from io import BytesIO
from pathlib import Path

app = typer.Typer()


@app.command()
def extract_data(
    call_audio_dir: Path = Path("/dataset/png_prod/call_audio"),
    call_meta_dir: Path = Path("/dataset/png_prod/call_metadata"),
    output_dir: Path = Path("./data"),
    dataset_name: str = "png_gcp_2jan",
    verbose: bool = False,
):
    from pydub import AudioSegment
    from .utils import ExtendedPath, asr_data_writer, strip_silence
    from lenses import lens

    call_asr_data: Path = output_dir / Path("asr_data")
    call_asr_data.mkdir(exist_ok=True, parents=True)

    def wav_event_generator(call_audio_dir):
        for wav_path in call_audio_dir.glob("**/*.wav"):
            if verbose:
                typer.echo(f"loading events for file {wav_path}")
            call_wav = AudioSegment.from_file_using_temporary_files(wav_path)
            rel_meta_path = wav_path.with_suffix(".json").relative_to(call_audio_dir)
            meta_path = call_meta_dir / rel_meta_path
            events = ExtendedPath(meta_path).read_json()
            yield call_wav, wav_path, events

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

    def compute_endtime(call_wav, state):
        for (i, st) in enumerate(state):
            start_time = st["AsrResult"]["Alternatives"][0].get("StartTime", 0)
            transcript = st["AsrResult"]["Alternatives"][0]["Transcript"]
            if i + 1 < len(state):
                end_time = state[i + 1]["AsrResult"]["Alternatives"][0]["StartTime"]
            else:
                end_time = call_wav.duration_seconds
            full_code_seg = call_wav[start_time * 1000 : end_time * 1000]
            code_seg = strip_silence(full_code_seg)
            code_fb = BytesIO()
            code_seg.export(code_fb, format="wav")
            code_wav = code_fb.getvalue()
            # only starting 1 min audio has reliable alignment ignore rest
            if start_time > 60:
                if verbose:
                    print(f'start time over 60 seconds of audio skipping.')
                break
            # only if some reasonable audio data is present yield it
            if code_seg.duration_seconds < 0.5:
                if verbose:
                    print(f'transcript chunk "{transcript}" contains no audio skipping.')
                continue
            yield transcript, code_seg.duration_seconds, code_wav

    def asr_data_generator(call_wav, call_wav_fname, events):
        call_wav_0, call_wav_1 = call_wav.split_to_mono()
        asr_events = lens["Events"].Each()["Event"].Filter(contains_asr)
        call_evs_0 = asr_events.Filter(channel(0)).collect()(events)
        # Ignoring agent channel events
        # call_evs_1 = asr_events.Filter(channel(1)).collect()(events)
        if verbose:
            typer.echo(f"processing data points on {call_wav_fname}")
        call_data_0 = compute_endtime(call_wav_0, call_evs_0)
        # Ignoring agent channel
        # call_data_1 = compute_endtime(call_wav_1, call_evs_1)
        return call_data_0  # chain(call_data_0, call_data_1)

    def generate_call_asr_data():
        full_asr_data = []
        total_duration = 0
        for wav, wav_path, ev in wav_event_generator(call_audio_dir):
            asr_data = asr_data_generator(wav, wav_path, ev)
            total_duration += wav.duration_seconds
            full_asr_data.append(asr_data)
        typer.echo(f"loaded {len(full_asr_data)} calls of duration {total_duration}s")
        n_dps = asr_data_writer(call_asr_data, dataset_name, chain(*full_asr_data))
        typer.echo(f"written {n_dps} data points")

    generate_call_asr_data()


def main():
    app()


if __name__ == "__main__":
    main()
