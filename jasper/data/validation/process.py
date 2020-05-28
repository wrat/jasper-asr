import json
import shutil
from pathlib import Path

import typer
from tqdm import tqdm

from ..utils import (
    alnum_to_asr_tokens,
    ExtendedPath,
    asr_manifest_reader,
    asr_manifest_writer,
    get_mongo_conn,
)

app = typer.Typer()


def preprocess_datapoint(
    idx, rel_root, sample, use_domain_asr, annotation_only, enable_plots
):
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    from pydub import AudioSegment
    from nemo.collections.asr.metrics import word_error_rate

    try:
        res = dict(sample)
        res["real_idx"] = idx
        audio_path = rel_root / Path(sample["audio_filepath"])
        res["audio_path"] = str(audio_path)
        if use_domain_asr:
            res["spoken"] = alnum_to_asr_tokens(res["text"])
        else:
            res["spoken"] = res["text"]
        res["utterance_id"] = audio_path.stem
        if not annotation_only:
            from jasper.client import transcriber_pretrained, transcriber_speller

            aud_seg = (
                AudioSegment.from_file_using_temporary_files(audio_path)
                .set_channels(1)
                .set_sample_width(2)
                .set_frame_rate(24000)
            )
            res["pretrained_asr"] = transcriber_pretrained(aud_seg.raw_data)
            res["pretrained_wer"] = word_error_rate(
                [res["text"]], [res["pretrained_asr"]]
            )
            if use_domain_asr:
                res["domain_asr"] = transcriber_speller(aud_seg.raw_data)
                res["domain_wer"] = word_error_rate(
                    [res["spoken"]], [res["pretrained_asr"]]
                )
        if enable_plots:
            wav_plot_path = (
                rel_root / Path("wav_plots") / Path(audio_path.name).with_suffix(".png")
            )
            if not wav_plot_path.exists():
                fig = plt.Figure()
                ax = fig.add_subplot()
                (y, sr) = librosa.load(audio_path)
                librosa.display.waveplot(y=y, sr=sr, ax=ax)
                with wav_plot_path.open("wb") as wav_plot_f:
                    fig.set_tight_layout(True)
                    fig.savefig(wav_plot_f, format="png", dpi=50)
                    # fig.close()
            res["plot_path"] = str(wav_plot_path)
        return res
    except BaseException as e:
        print(f'failed on {idx}: {sample["audio_filepath"]} with {e}')


@app.command()
def dump_validation_ui_data(
    data_manifest_path: Path = typer.Option(
        Path("./data/asr_data/call_alphanum/manifest.json"), show_default=True
    ),
    dump_path: Path = typer.Option(
        Path("./data/valiation_data/ui_dump.json"), show_default=True
    ),
    use_domain_asr: bool = True,
    annotation_only: bool = True,
    enable_plots: bool = True,
):
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    plot_dir = data_manifest_path.parent / Path("wav_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Using data manifest:{data_manifest_path}")
    with data_manifest_path.open("r") as pf:
        pnr_jsonl = pf.readlines()
        pnr_funcs = [
            partial(
                preprocess_datapoint,
                i,
                data_manifest_path.parent,
                json.loads(v),
                use_domain_asr,
                annotation_only,
                enable_plots,
            )
            for i, v in enumerate(pnr_jsonl)
        ]

        def exec_func(f):
            return f()

        with ThreadPoolExecutor() as exe:
            print("starting all preprocess tasks")
            pnr_data = filter(
                None,
                list(
                    tqdm(
                        exe.map(exec_func, pnr_funcs),
                        position=0,
                        leave=True,
                        total=len(pnr_funcs),
                    )
                ),
            )
    if annotation_only:
        result = list(pnr_data)
    else:
        wer_key = "domain_wer" if use_domain_asr else "pretrained_wer"
        result = sorted(pnr_data, key=lambda x: x[wer_key], reverse=True)
    ui_config = {
        "use_domain_asr": use_domain_asr,
        "data": result,
        "annotation_only": annotation_only,
        "enable_plots": enable_plots,
    }
    ExtendedPath(dump_path).write_json(ui_config)


@app.command()
def dump_corrections(dump_path: Path = Path("./data/valiation_data/corrections.json")):
    col = get_mongo_conn().test.asr_validation

    cursor_obj = col.find({"type": "correction"}, projection={"_id": False})
    corrections = [c for c in cursor_obj]
    ExtendedPath(dump_path).write_json(corrections)


@app.command()
def fill_unannotated(
    processed_data_path: Path = Path("./data/valiation_data/ui_dump.json"),
    corrections_path: Path = Path("./data/valiation_data/corrections.json"),
):
    processed_data = json.load(processed_data_path.open())
    corrections = json.load(corrections_path.open())
    annotated_codes = {c["code"] for c in corrections}
    all_codes = {c["gold_chars"] for c in processed_data}
    unann_codes = all_codes - annotated_codes
    mongo_conn = get_mongo_conn().test.asr_validation
    for c in unann_codes:
        mongo_conn.find_one_and_update(
            {"type": "correction", "code": c},
            {"$set": {"value": {"status": "Inaudible", "correction": ""}}},
            upsert=True,
        )


@app.command()
def update_corrections(
    data_manifest_path: Path = Path("./data/asr_data/call_alphanum/manifest.json"),
    corrections_path: Path = Path("./data/valiation_data/corrections.json"),
    skip_incorrect: bool = True,
):
    def correct_manifest(manifest_data_gen, corrections_path):
        corrections = json.load(corrections_path.open())
        correct_set = {
            c["code"] for c in corrections if c["value"]["status"] == "Correct"
        }
        # incorrect_set = {c["code"] for c in corrections if c["value"]["status"] == "Inaudible"}
        correction_map = {
            c["code"]: c["value"]["correction"]
            for c in corrections
            if c["value"]["status"] == "Incorrect"
        }
        # for d in manifest_data_gen:
        #     if d["chars"] in incorrect_set:
        #         d["audio_path"].unlink()
        renamed_set = set()
        for d in manifest_data_gen:
            if d["chars"] in correct_set:
                yield {
                    "audio_filepath": d["audio_filepath"],
                    "duration": d["duration"],
                    "text": d["text"],
                }
            elif d["chars"] in correction_map:
                correct_text = correction_map[d["chars"]]
                if skip_incorrect:
                    print(
                        f'skipping incorrect {d["audio_path"]} corrected to {correct_text}'
                    )
                else:
                    renamed_set.add(correct_text)
                    new_name = str(Path(correct_text).with_suffix(".wav"))
                    d["audio_path"].replace(d["audio_path"].with_name(new_name))
                    new_filepath = str(Path(d["audio_filepath"]).with_name(new_name))
                    yield {
                        "audio_filepath": new_filepath,
                        "duration": d["duration"],
                        "text": alnum_to_asr_tokens(correct_text),
                    }
            else:
                # don't delete if another correction points to an old file
                if d["chars"] not in renamed_set:
                    d["audio_path"].unlink()
                else:
                    print(f'skipping deletion of correction:{d["chars"]}')

    typer.echo(f"Using data manifest:{data_manifest_path}")
    dataset_dir = data_manifest_path.parent
    dataset_name = dataset_dir.name
    backup_dir = dataset_dir.with_name(dataset_name + ".bkp")
    if not backup_dir.exists():
        typer.echo(f"backing up to :{backup_dir}")
        shutil.copytree(str(dataset_dir), str(backup_dir))
    manifest_gen = asr_manifest_reader(data_manifest_path)
    corrected_manifest = correct_manifest(manifest_gen, corrections_path)
    new_data_manifest_path = data_manifest_path.with_name("manifest.new")
    asr_manifest_writer(new_data_manifest_path, corrected_manifest)
    new_data_manifest_path.replace(data_manifest_path)


@app.command()
def clear_mongo_corrections():
    delete = typer.confirm("are you sure you want to clear mongo collection it?")
    if delete:
        col = get_mongo_conn().test.asr_validation
        col.delete_many({"type": "correction"})
        typer.echo("deleted mongo collection.")
    typer.echo("Aborted")


def main():
    app()


if __name__ == "__main__":
    main()
