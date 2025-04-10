# TODO: The flash attention wheel image is private
FROM ghcr.io/silogen/rocm6.2-vllm0.6.3-flash-attn2.6.3-wheels:static AS final_wheels

FROM rocm/dev-ubuntu-22.04:6.2 AS final

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

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    rocthrust-dev \
    hipsparse-dev \
    hipblas-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    python -m pip install -U pip

# Install minio
RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc \
    --create-dirs \
    -o /minio-binaries/mc && \
    chown -hR ${USER_NAME} /minio-binaries/ && \
    chmod +x /minio-binaries/mc

ENV PATH="${PATH}:/minio-binaries/:/root/scripts/"
RUN pip install /opt/rocm/share/amd_smi

# Some of these are set before they are created, but that is fine
ENV LLVM_SYMBOLIZER_PATH=/opt/rocm/llvm/bin/llvm-symbolizer
ENV PATH=$PATH:/opt/rocm/bin:/libtorch/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/libtorch/lib
ENV CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/libtorch/include:/libtorch/include/torch/csrc/api/include:/opt/rocm/include

COPY --from=final_wheels /wheels/*.whl /libs/

RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 \
    && pip install --no-cache-dir transformers[tokenizers]==4.49.0 \
    && pip install --no-cache-dir --force-reinstall \
    'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-manylinux_2_24_x86_64.whl' \
    --no-deps \
    && pip install --no-cache-dir deepspeed tensorboard

RUN if ls /libs/*.whl; then \
        python3 -m pip uninstall -y flash-attn && \
        python3 -m pip install /libs/flash_attn*.whl --no-cache-dir; \
    fi
