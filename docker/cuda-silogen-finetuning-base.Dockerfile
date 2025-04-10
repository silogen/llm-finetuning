FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

ENV USER_NAME=user
ENV USER_ID=1000
ENV GROUP_NAME=silogen-sdx
ENV GROUP_ID=1006
ENV PATH="/root/.cargo/bin:${PATH}"

RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} && \
    useradd -m -g ${GROUP_ID} -u ${USER_ID} ${USER_NAME}

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y software-properties-common && \
    DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get install -y python3.10 python3.10-dev curl git && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && curl -sSL https://install.python-poetry.org | python3.10 - --preview \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y  \
    && pip3 install --upgrade requests setuptools wheel packaging \
    && ln -fs /usr/bin/python3.10 /usr/bin/python

# Install the finetuning dependencies.
# Also install bitsandbytes here, it keeps emitting warnings when running on CPU, which are not allowed in our automated
# tests. In any case, bitsandbytes currently seems to only support a GPU backend so only the GPU targets are relevant.
COPY packages/finetuning/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt \
    && pip install --no-cache-dir flash-attn --no-build-isolation \
    && pip install --no-cache-dir bitsandbytes==0.44.1 --no-deps \
    && pip install --no-cache-dir deepspeed tensorboard

# Use WORKDIR to set the working directory and avoid using separate RUN command for chown
WORKDIR /code

# Install minio
RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    -o /minio-binaries/mc && \
    chown -hR ${USER_NAME} /minio-binaries/ && \
    chmod +x /minio-binaries/mc

ENV PATH="${PATH}:/minio-binaries/"

RUN mkdir models/ && chown ${USER_NAME} models/

RUN chown ${USER_NAME} /code
