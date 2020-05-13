import os
# from pathlib import Path

import typer
import rpyc
from rpyc.utils.server import ThreadedServer
import nemo.collections.asr as nemo_asr

app = typer.Typer()


class ASRDataService(rpyc.Service):
    def get_data_loader(self):
        return nemo_asr.AudioToTextDataLayer


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
