FROM ghcr.io/silogen/rocm-silogen-finetuning-base:main

COPY . /finetuning

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/rocm6.2.4 -r /finetuning/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/rocm6.2.4 /finetuning

WORKDIR /workdir
