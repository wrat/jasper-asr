import typer
from pathlib import Path
from .utils import generate_dates

app = typer.Typer()


@app.command()
def export_conv_json(
    conv_src: Path = typer.Option(Path("./conv_data.json"), show_default=True),
    conv_dest: Path = typer.Option(Path("./data/conv_data.json"), show_default=True),
):
    from .utils import ExtendedPath

    conv_data = ExtendedPath(conv_src).read_json()

    conv_data["dates"] = generate_dates()

    ExtendedPath(conv_dest).write_json(conv_data)


def main():
    app()


if __name__ == "__main__":
    main()
