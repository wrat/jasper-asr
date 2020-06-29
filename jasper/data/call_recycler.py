import typer
from pathlib import Path
from enum import Enum


app = typer.Typer()


@app.command()
def export_all_logs(
    call_logs_file: Path = typer.Option(Path("./call_logs.yaml"), show_default=True),
    domain: str = typer.Option("sia-data.agaralabs.com", show_default=True),
):
    from .utils import get_mongo_conn
    from collections import defaultdict
    from ruamel.yaml import YAML

    yaml = YAML()
    mongo_coll = get_mongo_conn()
    caller_calls = defaultdict(lambda: [])
    for call in mongo_coll.find():
        sysid = call["SystemID"]
        call_uri = f"http://{domain}/calls/{sysid}"
        caller = call["Caller"]
        caller_calls[caller].append(call_uri)
    caller_list = []
    for caller in caller_calls:
        caller_list.append({"name": caller, "calls": caller_calls[caller]})
    output_yaml = {"users": caller_list}
    typer.echo(f"exporting call logs to yaml file at {call_logs_file}")
    with call_logs_file.open("w") as yf:
        yaml.dump(output_yaml, yf)


@app.command()
def export_calls_between(
    start_cid: str,
    end_cid: str,
    call_logs_file: Path = typer.Option(Path("./call_logs.yaml"), show_default=True),
    domain: str = typer.Option("sia-data.agaralabs.com", show_default=True),
    mongo_port: int = 27017,
):
    from collections import defaultdict
    from ruamel.yaml import YAML
    from .utils import get_mongo_conn

    yaml = YAML()
    mongo_coll = get_mongo_conn(port=mongo_port)
    start_meta = mongo_coll.find_one({"SystemID": start_cid})
    end_meta = mongo_coll.find_one({"SystemID": end_cid})

    caller_calls = defaultdict(lambda: [])
    call_query = mongo_coll.find(
        {
            "StartTS": {"$gte": start_meta["StartTS"]},
            "EndTS": {"$lte": end_meta["EndTS"]},
        }
    )
    for call in call_query:
        sysid = call["SystemID"]
        call_uri = f"http://{domain}/calls/{sysid}"
        caller = call["Caller"]
        caller_calls[caller].append(call_uri)
    caller_list = []
    for caller in caller_calls:
        caller_list.append({"name": caller, "calls": caller_calls[caller]})
    output_yaml = {"users": caller_list}
    typer.echo(f"exporting call logs to yaml file at {call_logs_file}")
    with call_logs_file.open("w") as yf:
        yaml.dump(output_yaml, yf)


@app.command()
def copy_metas(
    call_logs_file: Path = typer.Option(Path("./call_logs.yaml"), show_default=True),
    output_dir: Path = Path("./data"),
    meta_dir: Path = Path("/tmp/call_metas"),
):
    from lenses import lens
    from ruamel.yaml import YAML
    from urllib.parse import urlsplit
    from shutil import copy2

    yaml = YAML()
    call_logs = yaml.load(call_logs_file.read_text())

    call_meta_dir: Path = output_dir / Path("call_metas")
    call_meta_dir.mkdir(exist_ok=True, parents=True)
    meta_dir.mkdir(exist_ok=True, parents=True)

    def get_cid(uri):
        return Path(urlsplit(uri).path).stem

    def copy_meta(uri):
        cid = get_cid(uri)
        saved_meta_path = call_meta_dir / Path(f"{cid}.json")
        dest_meta_path = meta_dir / Path(f"{cid}.json")
        if not saved_meta_path.exists():
            print(f"{saved_meta_path} not found")
        copy2(saved_meta_path, dest_meta_path)

    def download_meta_audio():
        call_lens = lens["users"].Each()["calls"].Each()
        call_lens.modify(copy_meta)(call_logs)

    download_meta_audio()


class ExtractionType(str, Enum):
    flow = "flow"
    data = "data"


@app.command()
def analyze(
    leaderboard: bool = False,
    plot_calls: bool = False,
    extract_data: bool = False,
    extraction_type: ExtractionType = typer.Option(
        ExtractionType.data, show_default=True
    ),
    start_delay: float = 1.5,
    download_only: bool = False,
    strip_silent_chunks: bool = True,
    call_logs_file: Path = typer.Option(Path("./call_logs.yaml"), show_default=True),
    output_dir: Path = Path("./data"),
    data_name: str = None,
    mongo_uri: str = typer.Option(
        "mongodb://localhost:27017/test.calls", show_default=True
    ),
):

    from urllib.parse import urlsplit
    from functools import reduce
    import boto3
    from io import BytesIO
    import json
    from ruamel.yaml import YAML
    import re
    from google.protobuf.timestamp_pb2 import Timestamp
    from datetime import timedelta
    import librosa
    import librosa.display
    from lenses import lens
    from pprint import pprint
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    from tqdm import tqdm
    from .utils import ui_dump_manifest_writer, strip_silence, get_mongo_coll, get_call_logs
    from pydub import AudioSegment
    from natural.date import compress

    matplotlib.rcParams["agg.path.chunksize"] = 10000

    matplotlib.use("agg")

    yaml = YAML()
    s3 = boto3.client("s3")
    mongo_collection = get_mongo_coll(mongo_uri)
    call_media_dir: Path = output_dir / Path("call_wavs")
    call_media_dir.mkdir(exist_ok=True, parents=True)
    call_meta_dir: Path = output_dir / Path("call_metas")
    call_meta_dir.mkdir(exist_ok=True, parents=True)
    call_plot_dir: Path = output_dir / Path("plots")
    call_plot_dir.mkdir(exist_ok=True, parents=True)
    call_asr_data: Path = output_dir / Path("asr_data")
    call_asr_data.mkdir(exist_ok=True, parents=True)
    dataset_name = call_logs_file.stem if not data_name else data_name

    call_logs = yaml.load(call_logs_file.read_text())

    def gen_ev_fev_timedelta(fev):
        fev_p = Timestamp()
        fev_p.FromJsonString(fev["CreatedTS"])
        fev_dt = fev_p.ToDatetime()
        td_0 = timedelta()

        def get_timedelta(ev):
            ev_p = Timestamp()
            ev_p.FromJsonString(value=ev["CreatedTS"])
            ev_dt = ev_p.ToDatetime()
            delta = ev_dt - fev_dt
            return delta if delta > td_0 else td_0

        return get_timedelta

    def chunk_n(evs, n):
        return [evs[i * n : (i + 1) * n] for i in range((len(evs) + n - 1) // n)]

    if extraction_type == ExtractionType.data:

        def is_utter_event(ev):
            return (
                (ev["Author"] == "CONV" or ev["Author"] == "ASR")
                and (ev["Type"] != "DEBUG")
                and ev["Type"] != "ASR_RESULT"
            )

        def get_data_points(utter_events, td_fn):
            data_points = []
            for evs in chunk_n(utter_events, 3):
                try:
                    assert evs[0]["Type"] == "CONV_RESULT"
                    assert evs[1]["Type"] == "STARTED_SPEAKING"
                    assert evs[2]["Type"] == "STOPPED_SPEAKING"
                    start_time = td_fn(evs[1]).total_seconds() - start_delay
                    end_time = td_fn(evs[2]).total_seconds()
                    spoken = evs[0]["Msg"]
                    data_points.append(
                        {"start_time": start_time, "end_time": end_time, "code": spoken}
                    )
                except AssertionError:
                    # skipping invalid data_points
                    pass
            return data_points

        def text_extractor(spoken):
            return (
                re.search(r"'(.*)'", spoken).groups(0)[0]
                if len(spoken) > 6 and re.search(r"'(.*)'", spoken)
                else spoken
            )

    elif extraction_type == ExtractionType.flow:

        def is_final_asr_event_or_spoken(ev):
            pld = json.loads(ev["Payload"])
            return (
                pld["AsrResult"]["Results"][0]["IsFinal"]
                if ev["Type"] == "ASR_RESULT"
                else True
            )

        def is_utter_event(ev):
            return (
                ev["Author"] == "CONV"
                or (ev["Author"] == "ASR" and is_final_asr_event_or_spoken(ev))
            ) and (ev["Type"] != "DEBUG")

        def get_data_points(utter_events, td_fn):
            data_points = []
            for evs in chunk_n(utter_events, 4):
                try:
                    assert len(evs) == 4
                    assert evs[0]["Type"] == "CONV_RESULT"
                    assert evs[1]["Type"] == "STARTED_SPEAKING"
                    assert evs[2]["Type"] == "ASR_RESULT"
                    assert evs[3]["Type"] == "STOPPED_SPEAKING"
                    start_time = td_fn(evs[1]).total_seconds() - start_delay
                    end_time = td_fn(evs[2]).total_seconds()
                    conv_msg = evs[0]["Msg"]
                    if "full name" in conv_msg.lower():
                        pld = json.loads(evs[2]["Payload"])
                        spoken = pld["AsrResult"]["Results"][0]["Alternatives"][0][
                            "Transcript"
                        ]
                        data_points.append(
                            {
                                "start_time": start_time,
                                "end_time": end_time,
                                "code": spoken,
                            }
                        )
                except AssertionError:
                    # skipping invalid data_points
                    pass
            return data_points

        def text_extractor(spoken):
            return spoken

    def process_call(call_obj):
        call_meta = get_call_logs(call_obj, s3, call_meta_dir)
        call_events = call_meta["Events"]

        def is_writer_uri_event(ev):
            return ev["Author"] == "AUDIO_WRITER" and "s3://" in ev["Msg"]

        writer_events = list(filter(is_writer_uri_event, call_events))
        s3_wav_url = re.search(r"(s3://.*)", writer_events[0]["Msg"]).groups(0)[0]
        s3_wav_url_p = urlsplit(s3_wav_url)

        def is_first_audio_ev(state, ev):
            if state[0]:
                return state
            else:
                return (ev["Author"] == "GATEWAY" and ev["Type"] == "AUDIO", ev)

        (_, first_audio_ev) = reduce(is_first_audio_ev, call_events, (False, {}))

        get_ev_fev_timedelta = gen_ev_fev_timedelta(first_audio_ev)

        uevs = list(filter(is_utter_event, call_events))
        ev_count = len(uevs)
        utter_events = uevs[: ev_count - ev_count % 3]
        saved_wav_path = call_media_dir / Path(Path(s3_wav_url_p.path).name)
        if not saved_wav_path.exists():
            print(f"downloading : {saved_wav_path} from {s3_wav_url}")
            s3.download_file(
                s3_wav_url_p.netloc, s3_wav_url_p.path[1:], str(saved_wav_path)
            )

        return {
            "wav_path": saved_wav_path,
            "num_samples": len(utter_events) // 3,
            "meta": call_obj,
            "first_event_fn": get_ev_fev_timedelta,
            "utter_events": utter_events,
        }

    def get_cid(uri):
        return Path(urlsplit(uri).path).stem

    def ensure_call(uri):
        cid = get_cid(uri)
        meta = mongo_collection.find_one({"SystemID": cid})
        process_meta = process_call(meta)
        return process_meta

    def retrieve_processed_callmeta(uri):
        cid = get_cid(uri)
        meta = mongo_collection.find_one({"SystemID": cid})
        duration = meta["EndTS"] - meta["StartTS"]
        process_meta = process_call(meta)
        data_points = get_data_points(
            process_meta["utter_events"], process_meta["first_event_fn"]
        )
        process_meta["data_points"] = data_points
        return {"url": uri, "meta": meta, "duration": duration, "process": process_meta}

    def retrieve_callmeta(call_uri):
        uri = call_uri["call_uri"]
        name = call_uri["name"]
        cid = get_cid(uri)
        meta = mongo_collection.find_one({"SystemID": cid})
        duration = meta["EndTS"] - meta["StartTS"]
        process_meta = process_call(meta)
        data_points = get_data_points(
            process_meta["utter_events"], process_meta["first_event_fn"]
        )
        process_meta["data_points"] = data_points
        return {
            "url": uri,
            "name": name,
            "meta": meta,
            "duration": duration,
            "process": process_meta,
        }

    def download_meta_audio():
        call_lens = lens["users"].Each()["calls"].Each()
        call_lens.modify(ensure_call)(call_logs)

    def plot_calls_data():
        def plot_data_points(y, sr, data_points, file_path):
            plt.figure(figsize=(16, 12))
            librosa.display.waveplot(y=y, sr=sr)
            for dp in data_points:
                start, end, code = dp["start_time"], dp["end_time"], dp["code"]
                plt.axvspan(start, end, color="green", alpha=0.2)
                text_pos = (start + end) / 2
                plt.text(
                    text_pos,
                    0.25,
                    f"{code}",
                    rotation=90,
                    horizontalalignment="center",
                    verticalalignment="center",
                )
            plt.title("Datapoints")
            plt.savefig(file_path, format="png")
            return file_path

        def plot_call(call_obj):
            saved_wav_path, data_points, sys_id = (
                call_obj["process"]["wav_path"],
                call_obj["process"]["data_points"],
                call_obj["meta"]["SystemID"],
            )
            file_path = call_plot_dir / Path(sys_id).with_suffix(".png")
            if not file_path.exists():
                print(f"plotting: {file_path}")
                (y, sr) = librosa.load(saved_wav_path)
                plot_data_points(y, sr, data_points, str(file_path))
            return file_path

        call_lens = lens["users"].Each()["calls"].Each()
        call_stats = call_lens.modify(retrieve_processed_callmeta)(call_logs)
        # call_plot_data = call_lens.collect()(call_stats)
        call_plots = call_lens.modify(plot_call)(call_stats)
        # with ThreadPoolExecutor(max_workers=20) as exe:
        #     print('starting all plot tasks')
        #     responses = [exe.submit(plot_call, w) for w in call_plot_data]
        #     print('submitted all plot tasks')
        #     call_plots = [r.result() for r in responses]
        pprint(call_plots)

    def extract_data_points():
        if strip_silent_chunks:

            def audio_process(seg):
                return strip_silence(seg)

        else:

            def audio_process(seg):
                return seg

        def gen_data_values(saved_wav_path, data_points, caller_name):
            call_seg = (
                AudioSegment.from_wav(saved_wav_path)
                .set_channels(1)
                .set_sample_width(2)
                .set_frame_rate(24000)
            )
            for dp_id, dp in enumerate(data_points):
                start, end, spoken = dp["start_time"], dp["end_time"], dp["code"]
                spoken_seg = audio_process(call_seg[start * 1000 : end * 1000])
                spoken_fb = BytesIO()
                spoken_seg.export(spoken_fb, format="wav")
                spoken_wav = spoken_fb.getvalue()
                # search for actual pnr code and handle plain codes as well
                extracted_code = text_extractor(spoken)
                if strip_silent_chunks and spoken_seg.duration_seconds < 0.5:
                    print(f'transcript chunk "{spoken}" contains no audio skipping.')
                    continue
                yield extracted_code, spoken_seg.duration_seconds, spoken_wav, caller_name, spoken_seg

        call_lens = lens["users"].Each()["calls"].Each()

        def assign_user_call(uc):
            return (
                lens["calls"]
                .Each()
                .modify(lambda c: {"call_uri": c, "name": uc["name"]})(uc)
            )

        user_call_logs = lens["users"].Each().modify(assign_user_call)(call_logs)
        call_stats = call_lens.modify(retrieve_callmeta)(user_call_logs)
        call_objs = call_lens.collect()(call_stats)

        def data_source():
            for call_obj in tqdm(call_objs):
                saved_wav_path, data_points, name = (
                    call_obj["process"]["wav_path"],
                    call_obj["process"]["data_points"],
                    call_obj["name"],
                )
                for dp in gen_data_values(saved_wav_path, data_points, name):
                    yield dp

        ui_dump_manifest_writer(call_asr_data, dataset_name, data_source())

    def show_leaderboard():
        def compute_user_stats(call_stat):
            n_samples = (
                lens["calls"].Each()["process"]["num_samples"].get_monoid()(call_stat)
            )
            n_duration = lens["calls"].Each()["duration"].get_monoid()(call_stat)
            return {
                "num_samples": n_samples,
                "duration": n_duration.total_seconds(),
                "samples_rate": n_samples / n_duration.total_seconds(),
                "duration_str": compress(n_duration, pad=" "),
                "name": call_stat["name"],
            }

        call_lens = lens["users"].Each()["calls"].Each()
        call_stats = call_lens.modify(retrieve_processed_callmeta)(call_logs)
        user_stats = lens["users"].Each().modify(compute_user_stats)(call_stats)
        leader_df = (
            pd.DataFrame(user_stats["users"])
            .sort_values(by=["duration"], ascending=False)
            .reset_index(drop=True)
        )
        leader_df["rank"] = leader_df.index + 1
        leader_board = leader_df.rename(
            columns={
                "rank": "Rank",
                "num_samples": "Count",
                "name": "Name",
                "samples_rate": "SpeechRate",
                "duration_str": "Duration",
            }
        )[["Rank", "Name", "Count", "Duration"]]
        print(
            """ASR Dataset Leaderboard  :
---------------------------------"""
        )
        print(leader_board.to_string(index=False))

    if download_only:
        download_meta_audio()
        return
    if leaderboard:
        show_leaderboard()
    if plot_calls:
        plot_calls_data()
    if extract_data:
        extract_data_points()


def main():
    app()


if __name__ == "__main__":
    main()
