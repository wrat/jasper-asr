import typer
from pathlib import Path
from random import randrange
from itertools import product
from math import floor

app = typer.Typer()


@app.command()
def export_conv_json(
    conv_src: Path = typer.Option(Path("./conv_data.json"), show_default=True),
    conv_dest: Path = typer.Option(Path("./data/conv_data.json"), show_default=True),
):
    from .utils import ExtendedPath

    conv_data = ExtendedPath(conv_src).read_json()

    days = [i for i in range(1, 32)]
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    # ordinal from https://stackoverflow.com/questions/9647202/ordinal-numbers-replacement

    def ordinal(n):
        return "%d%s" % (
            n,
            "tsnrhtdd"[(floor(n / 10) % 10 != 1) * (n % 10 < 4) * n % 10 :: 4],
        )

    def canon_vars(d, m):
        return [
            ordinal(d) + " " + m,
            m + " " + ordinal(d),
            ordinal(d) + " of " + m,
            m + " the " + ordinal(d),
            str(d) + " " + m,
            m + " " + str(d),
        ]

    day_months = [dm for d, m in product(days, months) for dm in canon_vars(d, m)]

    conv_data["dates"] = day_months

    def dates_data_gen():
        i = randrange(len(day_months))
        return day_months[i]

    ExtendedPath(conv_dest).write_json(conv_data)


def main():
    app()


if __name__ == "__main__":
    main()
