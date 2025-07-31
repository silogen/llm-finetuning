FROM ghcr.io/silogen/rocm-silogen-finetuning-base:main

COPY . /finetuning

# Remove torch from requirements.txt to avoid installing a wrong version on top (custom install already included)
RUN sed -i '/torch.*/d' /finetuning/requirements.txt
RUN pip install --no-cache-dir -r /finetuning/requirements.txt
RUN pip install --no-cache-dir /finetuning

WORKDIR /workdir
