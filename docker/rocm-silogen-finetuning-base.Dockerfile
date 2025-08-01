FROM rocm/vllm:rocm6.4.1_vllm_0.9.1_20250715

ARG DEBIAN_FRONTEND=noninteractive

# This image is designed to be run as non-root user
ENV USER_NAME=user
ENV USER_ID=1000
ENV GROUP_NAME=silogen
ENV GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} && \
    useradd -m -g ${GROUP_ID} -u ${USER_ID} ${USER_NAME} \
    && mkdir /workdir \
    && chown -hR ${USER_NAME}:${GROUP_NAME} /workdir \
    && usermod -a -G video,render ${USER_NAME}  # Need to be in video and render to access GPUs

# Install minio
RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    -o /minio-binaries/mc && \
    chown -hR ${USER_NAME} /minio-binaries/ && \
    chmod +x /minio-binaries/mc

ENV PATH="${PATH}:/minio-binaries/:/root/scripts/"
RUN pip install /opt/rocm/share/amd_smi

RUN pip install --no-cache-dir transformers[tokenizers]==4.53.0 \
    && pip install --no-cache-dir --force-reinstall \
    'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-1.0.0-py3-none-manylinux_2_24_x86_64.whl' \
    --no-deps \
    && pip install --no-cache-dir deepspeed tensorboard
