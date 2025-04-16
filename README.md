# LLM Finetuning

This is the `finetuning` package, also known as the Silogen finetuning engine.

- 🤗 Built on top of HuggingFace libraries 🤗
- 🤝 Supports AMD and Nvidia GPUs 🤝
- 💪 Users write a YAML config file to configure and launch any LLM finetuning experiment 💪
- 🔎 Covered by a wide suite of tests 🔎

## Building the images

To split automatic builds into two parts, there are two Dockerfiles per platform (ROCm / CUDA), a base image that has the major dependencies, and a worker image that adds the package code and the final dependencies on top.
To build both the base and the worker locally and then push it to a container registry, run the script `docker/build-and-push-silogen-finetuning-image.sh`.

## Running finetuning

The finetuning package installs a script to run finetuning:

```
> finetuning --help
usage: finetuning [-h] [--logging-level LOGGING_LEVEL] [--num-preprocess-workers NUM_PREPROCESS_WORKERS]
                  [--mlflow-server-uri MLFLOW_SERVER_URI] [--experiment-name EXPERIMENT_NAME]
                  [--hf-mlflow-log-artifacts HF_MLFLOW_LOG_ARTIFACTS]
                  {sft,dpo} config

Finetuning

positional arguments:
  {sft,dpo}             The kind of finetuning to run. SFT is Supervised FineTuning. While DPO stands for Direct Preference Optimization,
                        note that it implements a set of related preference optimization algorithms such as IPO and KTO as well.
  config                Path to the experiment's YAML config file.

options:
  -h, --help            show this help message and exit
  --logging-level LOGGING_LEVEL
  --num-preprocess-workers NUM_PREPROCESS_WORKERS
                        Number of processes to use for preprocessing
  --mlflow-server-uri MLFLOW_SERVER_URI
                        MLflow server URI. Can be local path.
  --experiment-name EXPERIMENT_NAME
                        Experiment name that is used for MLflow tracking
  --hf-mlflow-log-artifacts HF_MLFLOW_LOG_ARTIFACTS
                        Whether to store model artifacts in MLFlow
```

## Automatic batch size configuration

The finetuning package sets the per-device batchsize semi-automatically. You define the target total batch size, and the maximum per-device batch size, and then you can start the same config with one training process or many parallel ones, and finetuning sets the per-device batch size as high as allowed, while keeping the total batch size at the target number.

Total batch size is the effective batch size for the complete training run. It is equal to `number of processes` \* `per-device batch size` \* `accumulation`.

The maximum batch size per device is the maximum batch size that can be accommodated on a single device. This mostly limited by the memory capacity of the device.

## Supervised Finetuning

Features:

- PEFT Adapters
- Expanding the vocabulary to add new special tokens (e.g. `<|im_start|>` and `<|im_end|>`)
- Completions-only training

## Preference Optimization

Features:

- DPO, plus variants, see [the HuggingFace
  options](https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOTrainer.loss_type)
- PEFT adapters for the DPO policy.

Notes:

- Merge the SFT PEFT Adapter before use! We had issues getting two adapters to run right (though HuggingFace should
  support this configuration too).

## Preparing models for inference

The finetuning methods install a wrapper for the tokenizer saving method, which sets the tokenizer up for inference when
it is saved.

The inference time setting are:

```
add_bos_token = True
add_eos_token = False
eos_token = chat_template.assistant_end  # e.g. '<|im_end|>'
```

Furthermore, for inference you must

- merge the adapter in to the model OR
- prepare vLLM compatible adapters

For **merging the adapter**, use the `merge_adapter` CLI.

```
> merge_adapter  --help
usage: merge_adapter [-h] [--tokenizer TOKENIZER] [--device_map DEVICE_MAP] basemodel peftmodel outpath

Merges adapters into a base model and saves the output as a single model

positional arguments:
  basemodel             Name (e.g. HuggingFace Hub id) or Path of Base Model
  peftmodel             Name or Path of Peft adapter
  outpath               Where to save the merged model

options:
  -h, --help            show this help message and exit
  --tokenizer TOKENIZER
                        Name or Path to tokenizer to save with the model. If not specified, will use the adapter model tokenizer.
  --device_map DEVICE_MAP
                        Device map for loading the model. Can be 'auto', 'cpu', 'cuda' or a JSON string.
```

For **preparing vLLM compatible adapters**, use the `create_vllm_compatible_adapter` CLI.

```
> create_vllm_compatible_adapter --help
usage: create_vllm_compatible_adapter [-h] [--training-config TRAINING_CONFIG] model_path

Take in a HuggingFace adapter's binary folder path and remove the embeddings layer weights required for vLLM compatibility

positional arguments:
  model_path            Path to the model folder (e.g. path to the folder containing the adapter) to be made compatible

options:
  -h, --help            show this help message and exit
  --training-config TRAINING_CONFIG
                        Path to training config to check and determine whether the layers should be removed or not.
```

## Setting up MLFlow logging

An example of tracking config part in sft config yaml file is:

```yaml
tracking:
  mlflow_server_uri: "file:///home/<USERNAME>/mlruns"
  experiment_name: "default"
  hf_mlflow_log_artifacts: "False"
```

One has two choices how to record experiment details to MLFlow. It can be local directory or a remote MLFlow instance.
If recording to local directory on your compute node instance you can set `tracking.mlflow_server_uri` to a local directory path
prefixed with `file://`, e.g. `file:///home/<USERNAME>/mlruns`. If you ran it on Docker, make sure that you map this directory to the host directory, so that all recorded information gets persisted after Docker container stops. Then you can launch mlflow ui using a command:

```bash
mlflow ui --backend-store-uri file:///home/<USERNAME>/mlruns
```

and do a port forward to your local machine:

```bash
ssh -L 5003:localhost:5000 YOUR_COMPUTE_NODE
```

Then MLFlow UI will be available in your browser at the address `http://127.0.0.1:5003`. Please make sure that `mlflow` package is
installed in your python environment.

If opting for a remote MLFlow server set MLFlow URI respectively.

If you want your experiment runs to be comparable in the MLFlow UI you should use the same `experiment_name` for all of them that you want to compare.

If you want to log artifacts generated by your trainer set `hf_mlflow_log_artifacts` to `True`. This only makes sense if logging to a remote server, e.g. s3 or GCS. If set to `True` or `1`, will copy each saved checkpoint on each save in TrainingArguments’s `output_dir` to the local or remote artifact storage. Using it without a remote storage will just copy the files to your artifact location.

`tracking` section of sft config is optional and can be skipped overall. `mlflow_server_uri` in `tracking` section of sft config is optional too and can be set empty to indicate that sft config is not used to set up MLFlow tracking. MLFlow configuration set through config file has precedence over settings supplied in environmental variables, when both ways of configuring MLFlow are used simultaneously. In turn, MLFlow configuration set through command line arguments has precedence over both config file and environment variables. Alternative way to activate MLFlow tracking would be to set environment variables e.g.:

```bash
export MLFLOW_EXPERIMENT_NAME=your_experiment_name
export MLFLOW_FLATTEN_PARAMS=TRUE
export MLFLOW_TRACKING_URI=file:///home/<USERNAME>/mlruns
export HF_MLFLOW_LOG_ARTIFACTS="FALSE"
```

If you want to disable MLFLow integration set environment variable:

`export DISABLE_MLFLOW_INTEGRATION=TRUE`

### Known issues

Logging large artifacts to a remote instance of MLflow running in a kubernetes cluster can fail with a `503` code. Likely we have to increase memory requests or limits for mlflow pod or setting `--gunicorn-opts="--timeout <LARGE VALUE>"` in the server launch command. Use `--hf-mlflow-log-artifacts=False` or `export HF_MLFLOW_LOG_ARTIFACTS="FALSE"`. Let us fix this when requested.

## Quantization

You can quantize the basemodel to reduce its memory consumption. This is helpful if you are training adapters. Typically
parameters are stored in bfloat16 which uses two bytes for each parameter. Thus a 34B parameter model takes about 68GB
in memory. This can be halved by using 8bit precision or quartered by using 4bit precision. Of course, you need much
more memory in training than just the basemodel parameters.

We currently use bitsandbytes, see configuration options
[here](https://huggingface.co/docs/transformers/v4.38.2/en/main_classes/quantization#transformers.BitsAndBytesConfig).
Quantization is enabled with a quantization section of the config, e.g. something like:

```yaml
quant_conf:
  quantization_type: "bits-and-bytes"
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: "nf4"
```

## Generate with HuggingFace

Although the evaluation package and the inference services are our current main evaluation and inference options, we
also allow HuggingFace-based generation through the finetuning package. The reason is primarily to enable us to use all
the HuggingFace options, which should match those available during training. For instance bitsandbytes quantization is
not supported by vLLM, but is supported by HuggingFace.

An example of running generation via docker run:

```bash
docker run \
  --rm --gpus --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
  -it \
  --mount type=bind,source="/local/silogen-sdx/experiments/,target=/experiments" \
  --mount type=bind,source="/local/silogen-sdx/models/,target=/models" \
  --mount type=bind,source="/local/silogen-sdx/mounts/HF_HOME/,target=/HF_HOME" \
  --mount type=bind,source="/local/silogen-sdx/datasets/,target=/datasets" \
  -e ACCELERATE_LOG_LEVEL=info \
  --entrypoint generate \
  rocm-silogen-finetuning-worker:YOUR_CHOSEN_TAG \
  /path/to/generate_config.yaml \
  /path/to/output.yaml
```

For an example of the generation config file see the top of `src/finetuning/generate.py`.

The output format is a YAML file that has at its root a list. Each member of that list has keys such as `prompt`,
`answer`, `correct_answers` similar to how tooling/inference structures its output.
