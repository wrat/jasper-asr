import typer
from pathlib import Path
import json

# from .utils import generate_dates, asr_test_writer

app = typer.Typer()


def run_test(reg_path, coll, s3, call_meta_dir, city_code, test_path):
    from time import sleep
    import subprocess
    from .utils import ExtendedPath, get_call_logs

    coll.delete_many({"CallID": test_path.name})
    # test_path = dump_dir / data_name / test_file
    # "../saas_reg/regression/run.sh  -f data/asr_data/call_upwork_test_cnd_cities/asr_test.reg"
    test_output = subprocess.run(
        ["/bin/bash", "-c", f"{str(reg_path)} --addr [::]:15400 -f {str(test_path)}"]
    )
    if test_output.returncode != 0:
        print("Error running test {test_file}")
        return

    def get_meta():
        call_meta = coll.find_one({"CallID": test_path.name})
        if call_meta:
            return call_meta
        else:
            sleep(2)
            return get_meta()

    call_meta = get_meta()
    call_logs = get_call_logs(call_meta, s3, call_meta_dir)
    call_events = call_logs["Events"]

    test_data_path = test_path.with_suffix(".result.json")
    test_data = ExtendedPath(test_data_path).read_json()

    def is_final_asr_event_or_spoken(ev):
        pld = json.loads(ev["Payload"])
        return (
            pld["AsrResult"]["Results"][0]["IsFinal"]
            if ev["Type"] == "ASR_RESULT"
            else False
        )

    def is_test_event(ev):
        return (
            ev["Author"] == "NLU"
            or (ev["Author"] == "ASR" and is_final_asr_event_or_spoken(ev))
        ) and (ev["Type"] != "DEBUG")

    test_evs = list(filter(is_test_event, call_events))
    if len(test_evs) == 2:
        try:
            asr_payload = test_evs[0]["Payload"]
            asr_result = json.loads(asr_payload)["AsrResult"]["Results"][0]
            alt_tscripts = [alt["Transcript"] for alt in asr_result["Alternatives"]]
            gcp_result = "|".join(alt_tscripts)
            entity_asr = asr_result["AsrDynamicResults"][0]["Candidate"]["Transcript"]
            nlu_payload = test_evs[1]["Payload"]
            nlu_result_payload = json.loads(nlu_payload)["NluResults"]
            entity = test_data[0]["entity"]
            text = test_data[0]["text"]
            audio_filepath = test_data[0]["audio_filepath"]
            pretrained_asr = test_data[0]["pretrained_asr"]
            nlu_entity = list(json.loads(nlu_result_payload)["Entities"].values())[0]
            asr_entity = city_code[entity] if entity in city_code else "UNKNOWN"
            entities_match = asr_entity == nlu_entity
            result = "Success" if entities_match else "Fail"
            return {
                "expected_entity": entity,
                "text": text,
                "audio_filepath": audio_filepath,
                "pretrained_asr": pretrained_asr,
                "entity_asr": entity_asr,
                "google_asr": gcp_result,
                "nlu_result": nlu_result_payload,
                "asr_entity": asr_entity,
                "nlu_entity": nlu_entity,
                "result": result,
            }
        except Exception:
            return {
                "expected_entity": test_data[0]["entity"],
                "text": test_data[0]["text"],
                "audio_filepath": test_data[0]["audio_filepath"],
                "pretrained_asr": test_data[0]["pretrained_asr"],
                "entity_asr": "",
                "google_asr": "",
                "nlu_result": "",
                "asr_entity": "",
                "nlu_entity": "",
                "result": "Error",
            }
    else:
        return {
            "expected_entity": test_data[0]["entity"],
            "text": test_data[0]["text"],
            "audio_filepath": test_data[0]["audio_filepath"],
            "pretrained_asr": test_data[0]["pretrained_asr"],
            "entity_asr": "",
            "google_asr": "",
            "nlu_result": "",
            "asr_entity": "",
            "nlu_entity": "",
            "result": "Empty",
        }


@app.command()
def evaluate_slu(
    # conv_src: Path = typer.Option(Path("./conv_data.json"), show_default=True),
    data_name: str = typer.Option("call_upwork_test_cnd_cities", show_default=True),
    # extraction_key: str = "Cities",
    dump_dir: Path = Path("./data/asr_data"),
    call_meta_dir: Path = Path("./data/call_metas"),
    test_file_pref: str = "asr_test",
    mongo_uri: str = typer.Option(
        "mongodb://localhost:27017/test.calls", show_default=True
    ),
    test_results: Path = Path("./data/results.csv"),
    airport_codes: Path = Path("./airports_code.csv"),
    reg_path: Path = Path("../saas_reg/regression/run.sh"),
    test_id: str = "5ef481f27031edf6910e94e0",
):
    # import json
    from .utils import get_mongo_coll
    import pandas as pd
    import boto3
    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    # import subprocess
    # from time import sleep
    import csv
    from tqdm import tqdm

    s3 = boto3.client("s3")
    df = pd.read_csv(airport_codes)[["iata", "city"]]
    city_code = pd.Series(df["iata"].values, index=df["city"]).to_dict()

    test_files = list((dump_dir / data_name).glob(test_file_pref + "*.reg"))
    coll = get_mongo_coll(mongo_uri)
    with test_results.open("w") as csvfile:
        fieldnames = [
            "expected_entity",
            "text",
            "audio_filepath",
            "pretrained_asr",
            "entity_asr",
            "google_asr",
            "nlu_result",
            "asr_entity",
            "nlu_entity",
            "result",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=8) as exe:
            print("starting all loading tasks")
            for test_result in tqdm(
                exe.map(
                    partial(run_test, reg_path, coll, s3, call_meta_dir, city_code),
                    test_files,
                ),
                position=0,
                leave=True,
                total=len(test_files),
            ):
                writer.writerow(test_result)


def main():
    app()


if __name__ == "__main__":
    main()
