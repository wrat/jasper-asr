# import argparse

# import logging
import typer
from pathlib import Path

app = typer.Typer()
# leader_app = typer.Typer()
# app.add_typer(leader_app, name="leaderboard")
# plot_app = typer.Typer()
# app.add_typer(plot_app, name="plot")


@app.command()
def analyze(
    leaderboard: bool = False,
    plot_calls: bool = False,
    extract_data: bool = False,
    call_logs_file: Path = Path("./call_logs.yaml"),
    output_dir: Path = Path("./data"),
):
    call_logs_file = Path("./call_logs.yaml")
    output_dir = Path("./data")

    from urllib.parse import urlsplit
    from functools import reduce
    from pymongo import MongoClient
    import boto3

    from io import BytesIO
    import json
    from ruamel.yaml import YAML
    import re
    from google.protobuf.timestamp_pb2 import Timestamp
    from datetime import timedelta

    # from concurrent.futures import ThreadPoolExecutor
    from dateutil.relativedelta import relativedelta
    import librosa
    import librosa.display
    from lenses import lens
    from pprint import pprint
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    from tqdm import tqdm
    from .utils import asr_data_writer
    from pydub import AudioSegment
    # from itertools import product, chain

    matplotlib.rcParams["agg.path.chunksize"] = 10000

    matplotlib.use("agg")

    # logging.basicConfig(
    #     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    # )
    # logger = logging.getLogger(__name__)
    yaml = YAML()
    s3 = boto3.client("s3")
    mongo_collection = MongoClient("mongodb://localhost:27017/").test.calls
    call_media_dir: Path = output_dir / Path("call_wavs")
    call_media_dir.mkdir(exist_ok=True, parents=True)
    call_meta_dir: Path = output_dir / Path("call_metas")
    call_meta_dir.mkdir(exist_ok=True, parents=True)
    call_plot_dir: Path = output_dir / Path("plots")
    call_plot_dir.mkdir(exist_ok=True, parents=True)
    call_asr_data: Path = output_dir / Path("asr_data")
    call_asr_data.mkdir(exist_ok=True, parents=True)

    call_logs = yaml.load(call_logs_file.read_text())

    def get_call_meta(call_obj):
        s3_event_url_p = urlsplit(call_obj["DataURI"])
        saved_meta_path = call_meta_dir / Path(Path(s3_event_url_p.path).name)
        if not saved_meta_path.exists():
            print(f"downloading : {saved_meta_path}")
            s3.download_file(
                s3_event_url_p.netloc, s3_event_url_p.path[1:], str(saved_meta_path)
            )
        call_metas = json.load(saved_meta_path.open())
        return call_metas

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

    def process_call(call_obj):
        call_meta = get_call_meta(call_obj)
        call_events = call_meta["Events"]

        def is_writer_event(ev):
            return ev["Author"] == "AUDIO_WRITER"

        writer_events = list(filter(is_writer_event, call_events))
        s3_wav_url = re.search(r"saved to: (.*)", writer_events[0]["Msg"]).groups(0)[0]
        s3_wav_url_p = urlsplit(s3_wav_url)

        def is_first_audio_ev(state, ev):
            if state[0]:
                return state
            else:
                return (ev["Author"] == "GATEWAY" and ev["Type"] == "AUDIO", ev)

        (_, first_audio_ev) = reduce(is_first_audio_ev, call_events, (False, {}))

        get_ev_fev_timedelta = gen_ev_fev_timedelta(first_audio_ev)

        def is_utter_event(ev):
            return (
                (ev["Author"] == "CONV" or ev["Author"] == "ASR")
                and (ev["Type"] != "DEBUG")
                and ev["Type"] != "ASR_RESULT"
            )

        uevs = list(filter(is_utter_event, call_events))
        ev_count = len(uevs)
        utter_events = uevs[: ev_count - ev_count % 3]
        saved_wav_path = call_media_dir / Path(Path(s3_wav_url_p.path).name)
        if not saved_wav_path.exists():
            print(f"downloading : {saved_wav_path}")
            s3.download_file(
                s3_wav_url_p.netloc, s3_wav_url_p.path[1:], str(saved_wav_path)
            )

        # %config InlineBackend.figure_format = "retina"
        def chunk_n(evs, n):
            return [evs[i * n : (i + 1) * n] for i in range((len(evs) + n - 1) // n)]

        def get_data_points(utter_events):
            data_points = []
            for evs in chunk_n(utter_events, 3):
                assert evs[0]["Type"] == "CONV_RESULT"
                assert evs[1]["Type"] == "STARTED_SPEAKING"
                assert evs[2]["Type"] == "STOPPED_SPEAKING"
                start_time = get_ev_fev_timedelta(evs[1]).total_seconds() - 1.5
                end_time = get_ev_fev_timedelta(evs[2]).total_seconds()
                code = evs[0]["Msg"]
                data_points.append(
                    {"start_time": start_time, "end_time": end_time, "code": code}
                )
            return data_points

        def plot_events(y, sr, utter_events, file_path):
            plt.figure(figsize=(16, 12))
            librosa.display.waveplot(y=y, sr=sr)
            # plt.tight_layout()
            for evs in chunk_n(utter_events, 3):
                assert evs[0]["Type"] == "CONV_RESULT"
                assert evs[1]["Type"] == "STARTED_SPEAKING"
                assert evs[2]["Type"] == "STOPPED_SPEAKING"
                for ev in evs:
                    # print(ev["Type"])
                    ev_type = ev["Type"]
                    pos = get_ev_fev_timedelta(ev).total_seconds()
                    if ev_type == "STARTED_SPEAKING":
                        pos = pos - 1.5
                    plt.axvline(pos)  # , label="pyplot vertical line")
                    plt.text(
                        pos,
                        0.2,
                        f"event:{ev_type}:{ev['Msg']}",
                        rotation=90,
                        horizontalalignment="left"
                        if ev_type != "STOPPED_SPEAKING"
                        else "right",
                        verticalalignment="center",
                    )
            plt.title("Monophonic")
            plt.savefig(file_path, format="png")

        data_points = get_data_points(utter_events)

        return {
            "wav_path": saved_wav_path,
            "num_samples": len(utter_events) // 3,
            "meta": call_obj,
            "data_points": data_points,
        }

    def retrieve_callmeta(uri):
        cid = Path(urlsplit(uri).path).stem
        meta = mongo_collection.find_one({"SystemID": cid})
        duration = meta["EndTS"] - meta["StartTS"]
        process_meta = process_call(meta)
        return {"url": uri, "meta": meta, "duration": duration, "process": process_meta}

    # @plot_app.command()
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

        # plot_call(retrieve_callmeta("http://saasdev.agaralabs.com/calls/JOR9V47L03AGUEL"))
        call_lens = lens["users"].Each()["calls"].Each()
        call_stats = call_lens.modify(retrieve_callmeta)(call_logs)
        # call_plot_data = call_lens.collect()(call_stats)
        call_plots = call_lens.modify(plot_call)(call_stats)
        # with ThreadPoolExecutor(max_workers=20) as exe:
        #     print('starting all plot tasks')
        #     responses = [exe.submit(plot_call, w) for w in call_plot_data]
        #     print('submitted all plot tasks')
        #     call_plots = [r.result() for r in responses]
        pprint(call_plots)

    def extract_data_points():
        def gen_data_values(saved_wav_path, data_points):
            call_seg = (
                AudioSegment.from_wav(saved_wav_path)
                .set_channels(1)
                .set_sample_width(2)
                .set_frame_rate(24000)
            )
            for dp_id, dp in enumerate(data_points):
                start, end, code = dp["start_time"], dp["end_time"], dp["code"]
                code_seg = call_seg[start * 1000 : end * 1000]
                code_fb = BytesIO()
                code_seg.export(code_fb, format="wav")
                code_wav = code_fb.getvalue()
                # import pdb; pdb.set_trace()
                yield code, code_seg.duration_seconds, code_wav

        call_lens = lens["users"].Each()["calls"].Each()
        call_stats = call_lens.modify(retrieve_callmeta)(call_logs)
        call_objs = call_lens.collect()(call_stats)

        def data_source():
            for call_obj in tqdm(call_objs):
                saved_wav_path, data_points, sys_id = (
                    call_obj["process"]["wav_path"],
                    call_obj["process"]["data_points"],
                    call_obj["meta"]["SystemID"],
                )
                for dp in gen_data_values(saved_wav_path, data_points):
                    yield dp

        asr_data_writer(call_asr_data, "call_alphanum", data_source())

    # @leader_app.command()
    def show_leaderboard():
        def compute_user_stats(call_stat):
            n_samples = (
                lens["calls"].Each()["process"]["num_samples"].get_monoid()(call_stat)
            )
            n_duration = lens["calls"].Each()["duration"].get_monoid()(call_stat)
            rel_dur = relativedelta(
                seconds=int(n_duration.total_seconds()),
                microseconds=n_duration.microseconds,
            )
            return {
                "num_samples": n_samples,
                "duration": n_duration.total_seconds(),
                "samples_rate": n_samples / n_duration.total_seconds(),
                "duration_str": f"{rel_dur.minutes} mins {rel_dur.seconds} secs",
                "name": call_stat["name"],
            }

        call_lens = lens["users"].Each()["calls"].Each()
        call_stats = call_lens.modify(retrieve_callmeta)(call_logs)
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
                "num_samples": "Codes",
                "name": "Name",
                "samples_rate": "SpeechRate",
                "duration_str": "Duration",
            }
        )[["Rank", "Name", "Codes", "Duration"]]
        print(
            """Today's ASR Speller Dataset Leaderboard:
----------------------------------------"""
        )
        print(leader_board.to_string(index=False))

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
