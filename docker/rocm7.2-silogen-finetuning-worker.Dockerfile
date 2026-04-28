FROM ghcr.io/silogen/rocm7.2-silogen-finetuning-base:main

COPY . /finetuning

# Use a custom requirements.txt
RUN mv /finetuning/requirements-rocm7.2.txt /finetuning/requirements.txt

RUN sudo apt remove --yes python3-blinker \
    && pip install --no-cache-dir -r /finetuning/requirements.txt
RUN pip install --no-cache-dir /finetuning

WORKDIR /workdir
RUN mkdir /workdir/logs \
    && chown -hR ${USER_NAME}:${GROUP_NAME} /workdir/logs
