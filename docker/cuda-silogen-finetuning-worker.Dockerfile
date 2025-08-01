# Use the base image
FROM ghcr.io/silogen/cuda-silogen-finetuning-base:main

# Set working directory
WORKDIR /code

COPY . ./finetuning

# Environment variables for HuggingFace
ENV HF_HOME="/HF_HOME/"
ENV HF_HUB_OFFLINE=1

# Create and set permissions for the HuggingFace cache directory
RUN mkdir -p ${HF_HOME} && chown ${USER_NAME}:${GROUP_NAME} ${HF_HOME}

# Create and own directories for the basemodel, datasets, and the run directory:
RUN mkdir /rundir && chown ${USER_NAME}:${GROUP_NAME} /rundir \
    && mkdir /datasets && chown ${USER_NAME}:${GROUP_NAME} /datasets \
    && mkdir /basemodel && chown ${USER_NAME}:${GROUP_NAME} /basemodel

# Set the entrypoint and command
ENTRYPOINT [ "finetuning" ]
CMD [ "--help" ]
