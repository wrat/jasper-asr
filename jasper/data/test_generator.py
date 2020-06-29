import typer
from pathlib import Path
from .utils import generate_dates, asr_test_writer

app = typer.Typer()


@app.command()
def export_test_reg(
    conv_src: Path = typer.Option(Path("./conv_data.json"), show_default=True),
    data_name: str = typer.Option("call_upwork_test_cnd_cities", show_default=True),
    extraction_key: str = "Cities",
    dump_dir: Path = Path("./data/asr_data"),
    dump_file: Path = Path("ui_dump.json"),
    manifest_file: Path = Path("manifest.json"),
    test_file: Path = Path("asr_test.reg"),
):
    from .utils import (
        ExtendedPath,
        asr_manifest_reader,
        gcp_transcribe_gen,
        parallel_apply,
    )
    from ..client import transcribe_gen
    from pydub import AudioSegment
    from queue import PriorityQueue

    jasper_map = {
        "PNRs": 8045,
        "Cities": 8046,
        "Names": 8047,
        "Dates": 8048,
    }
    # jasper_map = {"PNRs": 8050, "Cities": 8050, "Names": 8050, "Dates": 8050}
    transcriber_gcp = gcp_transcribe_gen()
    transcriber_trained = transcribe_gen(asr_port=jasper_map[extraction_key])
    transcriber_all_trained = transcribe_gen(asr_port=8050)
    transcriber_libri_all_trained = transcribe_gen(asr_port=8051)

    def find_ent(dd, conv_data):
        ents = PriorityQueue()
        for ent in conv_data:
            if ent in dd["text"]:
                ents.put((-len(ent), ent))
        return ents.get_nowait()[1]

    def process_data(d):
        orig_seg = AudioSegment.from_wav(d["audio_path"])
        jas_seg = orig_seg.set_channels(1).set_sample_width(2).set_frame_rate(24000)
        gcp_seg = orig_seg.set_channels(1).set_sample_width(2).set_frame_rate(16000)
        deepgram_file = Path("/home/shubham/voice_auto/pnrs/wav/") / Path(
            d["audio_path"].stem + ".txt"
        )
        if deepgram_file.exists():
            d["deepgram"] = "".join(
                [s.replace("CHANNEL 0:", "") for s in deepgram_file.read_text().split("\n")]
            )
        else:
            d["deepgram"] = 'Not Found'
        d["audio_path"] = str(d["audio_path"])
        d["gcp_transcript"] = transcriber_gcp(gcp_seg.raw_data)
        d["jasper_trained"] = transcriber_trained(jas_seg.raw_data)
        d["jasper_all"] = transcriber_all_trained(jas_seg.raw_data)
        d["jasper_libri"] = transcriber_libri_all_trained(jas_seg.raw_data)
        return d

    conv_data = ExtendedPath(conv_src).read_json()
    conv_data["Dates"] = generate_dates()

    dump_data_path = dump_dir / Path(data_name) / dump_file
    ui_dump_data = ExtendedPath(dump_data_path).read_json()["data"]
    ui_dump_map = {i["utterance_id"]: i for i in ui_dump_data}
    manifest_path = dump_dir / Path(data_name) / manifest_file
    test_points = list(asr_manifest_reader(manifest_path))
    test_data_objs = [{**(ui_dump_map[t["audio_path"].stem]), **t} for t in test_points]
    test_data = parallel_apply(process_data, test_data_objs)
    # test_data = [process_data(t) for t in test_data_objs]
    test_path = dump_dir / Path(data_name) / test_file

    def dd_gen(dump_data):
        for dd in dump_data:
            ent = find_ent(dd, conv_data[extraction_key])
            dd["entity"] = ent
            if ent:
                yield dd

    asr_test_writer(test_path, dd_gen(test_data))
    # for i, b in enumerate(batch(test_data, 1)):
    #     test_fname = Path(f"{test_file.stem}_{i}.reg")
    #     test_path = dump_dir / Path(data_name) / test_fname
    #     asr_test_writer(test_path, dd_gen(test_data))


def main():
    app()


if __name__ == "__main__":
    main()
