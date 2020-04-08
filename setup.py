from setuptools import setup

requirements = [
    "ruamel.yaml",
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@09e3ba4dfe333f86d6c5c1048e07210924294be9#egg=nemo_toolkit",
]

extra_requirements = {
    "server": ["rpyc~=4.1.4"],
    "data": [
        "google-cloud-texttospeech~=1.0.1",
        "tqdm~=4.39.0",
        "pydub~=0.23.1",
        "scikit_learn~=0.22.1",
        "pandas~=1.0.3",
        "boto3~=1.12.35",
    ],
}

setup(
    name="jasper-asr",
    version="0.1",
    description="Tool to get gcp alignments of tts-data",
    url="http://github.com/malarinv/jasper-asr",
    author="Malar Kannan",
    author_email="malarkannan.invention@gmail.com",
    license="MIT",
    install_requires=requirements,
    extras_require=extra_requirements,
    packages=["."],
    entry_points={
        "console_scripts": [
            "jasper_transcribe = jasper.transcribe:main",
            "jasper_asr_rpyc_server = jasper.server:main",
            "jasper_asr_trainer = jasper.train:main",
            "jasper_asr_data_generate = jasper.data_utils.generator:main",
        ]
    },
    zip_safe=False,
)
