import typer
import rpyc
import os
from pathlib import Path
from rpyc.utils.server import ThreadedServer

app = typer.Typer()


class ASRDataService(rpyc.Service):
    def get_data_loader(self, data_manifest: Path):
        return "hello"


@app.command()
def run_server(port: int = 0):
    listen_port = port if port else int(os.environ.get("ASR_RPYC_PORT", "8044"))
    service = ASRDataService()
    t = ThreadedServer(service, port=listen_port)
    typer.echo(f"starting asr server on {listen_port}...")
    t.start()


def main():
    app()


if __name__ == "__main__":
    main()
