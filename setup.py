from setuptools import setup, find_packages

requirements = [
    "ruamel.yaml",
    "torch==1.4.0",
    "torchvision==0.5.0",
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
        "ruamel.yaml==0.16.10",
        "pymongo==3.10.1",
        "librosa==0.7.2",
        "matplotlib==3.2.1",
        "pandas==1.0.3",
        "tabulate==0.8.7",
        "natural==0.2.0",
        "num2words==0.5.10",
        "typer[all]==0.1.1",
        "python-slugify==4.0.0",
        "lenses @ git+https://github.com/ingolemo/python-lenses.git@b2a2a9aa5b61540992d70b2cf36008d0121e8948#egg=lenses",
    ],
    "validation": [
        "rpyc~=4.1.4",
        "pymongo==3.10.1",
        "typer[all]==0.1.1",
        "tqdm~=4.39.0",
        "librosa==0.7.2",
        "matplotlib==3.2.1",
        "pydub~=0.23.1",
        "streamlit==0.58.0",
        "stringcase==1.2.0"
    ]
    # "train": [
    #     "torchaudio==0.5.0",
    #     "torch-stft==0.1.4",
    # ]
}
packages = find_packages()

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
    packages=packages,
    entry_points={
        "console_scripts": [
            "jasper_transcribe = jasper.transcribe:main",
            "jasper_asr_rpyc_server = jasper.server:main",
            "jasper_asr_trainer = jasper.training_utils.train:main",
            "jasper_asr_data_generate = jasper.data_utils.generator:main",
            "jasper_asr_data_recycle = jasper.data_utils.call_recycler:main",
            "jasper_asr_data_validation = jasper.data_utils.validation.process:main",
            "jasper_asr_data_preprocess = jasper.data_utils.process:main",
        ]
    },
    zip_safe=False,
)
