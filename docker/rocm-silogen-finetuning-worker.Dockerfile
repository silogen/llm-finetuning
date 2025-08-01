FROM ghcr.io/silogen/rocm-silogen-finetuning-base:main

COPY . /finetuning

# Remove torch from requirements.txt to avoid installing a wrong version on top (custom install already included)
RUN sed -i '/torch.*/d' /finetuning/requirements.txt
RUN sudo apt remove --yes python3-blinker \
    && pip install --no-cache-dir -r /finetuning/requirements.txt
RUN pip install --no-cache-dir /finetuning

WORKDIR /workdir
RUN mkdir /workdir/logs \
    && chown -hR ${USER_NAME}:${GROUP_NAME} /workdir/logs
