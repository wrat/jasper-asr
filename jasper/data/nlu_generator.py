from pathlib import Path

import typer
import pandas as pd
from ruamel.yaml import YAML
from itertools import product
from .utils import generate_dates

app = typer.Typer()


def unique_entity_list(entity_template_tags, entity_data):
    unique_entity_set = {
        t
        for n in range(1, 5)
        for t in entity_data[f"Answer.utterance-{n}"].tolist()
        if any(et in t for et in entity_template_tags)
    }
    return list(unique_entity_set)


def nlu_entity_reader(nlu_data_file: Path = Path("./nlu_data.yaml")):
    yaml = YAML()
    nlu_data = yaml.load(nlu_data_file.read_text())
    for cf in nlu_data["csv_files"]:
        data = pd.read_csv(cf["fname"])
        for et in cf["entities"]:
            entity_name = et["name"]
            entity_template_tags = et["tags"]
            if "filter" in et:
                entity_data = data[data[cf["filter_key"]] == et["filter"]]
            else:
                entity_data = data
            yield entity_name, entity_template_tags, entity_data


def nlu_samples_reader(nlu_data_file: Path = Path("./nlu_data.yaml")):
    yaml = YAML()
    nlu_data = yaml.load(nlu_data_file.read_text())
    sm = {s["name"]: s for s in nlu_data["samples_per_entity"]}
    return sm


@app.command()
def compute_unique_nlu_stats(
    nlu_data_file: Path = typer.Option(Path("./nlu_data.yaml"), show_default=True),
):
    for entity_name, entity_template_tags, entity_data in nlu_entity_reader(
        nlu_data_file
    ):
        entity_count = len(unique_entity_list(entity_template_tags, entity_data))
        print(f"{entity_name}\t{entity_count}")


def replace_entity(tmpl, value, tags):
    result = tmpl
    for t in tags:
        result = result.replace(t, value)
    return result


@app.command()
def export_nlu_conv_json(
    conv_src: Path = typer.Option(Path("./conv_data.json"), show_default=True),
    conv_dest: Path = typer.Option(Path("./data/conv_data.json"), show_default=True),
    nlu_data_file: Path = typer.Option(Path("./nlu_data.yaml"), show_default=True),
):
    from .utils import ExtendedPath
    from random import sample

    entity_samples = nlu_samples_reader(nlu_data_file)
    conv_data = ExtendedPath(conv_src).read_json()
    conv_data["Dates"] = generate_dates()
    result_dict = {}
    data_count = 0
    for entity_name, entity_template_tags, entity_data in nlu_entity_reader(
        nlu_data_file
    ):
        entity_variants = sample(conv_data[entity_name], entity_samples[entity_name]["test_size"])
        unique_entites = unique_entity_list(entity_template_tags, entity_data)
        # sample_entites = sample(unique_entites, entity_samples[entity_name]["samples"])
        result_dict[entity_name] = []
        for val in entity_variants:
            sample_entites = sample(unique_entites, entity_samples[entity_name]["samples"])
            for tmpl in sample_entites:
                result = replace_entity(tmpl, val, entity_template_tags)
                result_dict[entity_name].append(result)
                data_count += 1
    print(f"Total of {data_count} variants generated")
    ExtendedPath(conv_dest).write_json(result_dict)


def main():
    app()


if __name__ == "__main__":
    main()
