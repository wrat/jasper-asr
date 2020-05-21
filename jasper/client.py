import os
import logging
import rpyc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


ASR_HOST = os.environ.get("JASPER_ASR_RPYC_HOST", "localhost")
ASR_PORT = int(os.environ.get("JASPER_ASR_RPYC_PORT", "8045"))


def transcribe_gen(asr_host=ASR_HOST, asr_port=ASR_PORT):
    logger.info(f"connecting to asr server at {asr_host}:{asr_port}")
    asr = rpyc.connect(asr_host, asr_port).root
    logger.info(f"connected to asr server successfully")
    return asr.transcribe


transcriber_pretrained = transcribe_gen(asr_port=8044)
transcriber_speller = transcribe_gen(asr_port=8045)
