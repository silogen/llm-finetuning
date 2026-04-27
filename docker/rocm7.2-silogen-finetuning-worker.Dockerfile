FROM ghcr.io/silogen/rocm-silogen-finetuning-base:main

COPY . /finetuning

# Specify rocm versions for certain dependencies that exist in the base:
RUN sed -i -e '/torch.*/d' -e '/triton.*/d' /finetuning/requirements.txt \
    && echo "torch==2.9.1+rocm7.2.0.lw.git7e1940d4" >> /finetuning/requirements.txt \
    && echo "triton==3.5.1+rocm7.2.0.gita272dfa8" >> /finetuning/requirements.txt

RUN sudo apt remove --yes python3-blinker \
    && pip install --no-cache-dir -r /finetuning/requirements.txt
RUN pip install --no-cache-dir /finetuning

WORKDIR /workdir
RUN mkdir /workdir/logs \
    && chown -hR ${USER_NAME}:${GROUP_NAME} /workdir/logs
