FROM ghcr.io/silogen/rocm-silogen-finetuning-base:main

COPY packages/finetuning /finetuning

RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/rocm6.2 -r /finetuning/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/rocm6.2 /finetuning

WORKDIR /workdir
RUN mkdir /workdir/logs
