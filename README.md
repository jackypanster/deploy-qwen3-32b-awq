# Deploying Qwen3-32B-AWQ with vLLM

This document outlines the steps to deploy the `Qwen/Qwen3-32B-AWQ` model using vLLM, adhering to the requirements specified in `prd.md`.

## Prerequisites

- Docker installed.
- NVIDIA drivers installed on the host machine (as per `prd.md`: Driver Version 570.153.02, CUDA 12.8 compatible).
- Access to 4 NVIDIA GPUs with at least 22GB VRAM each.
- The `qwen3_nonthinking.jinja` chat template file in this directory.

## Deployment Command

To start the vLLM server with the OpenAI-compatible API, run the following command from the root of this workspace (`/home/user/workspace/deploy-qwen3-32b-awq`):

```bash
docker run -d \
  --runtime=nvidia \
  --gpus=all \
  --name coder \
  -v /home/llm/model/qwen/Qwen3-32B-AWQ:/model/Qwen3-32B-AWQ \
  -v /home/llm/model/qwen/Qwen3-32B-AWQ/qwen3_nonthinking.jinja:/app/qwen3_nonthinking.jinja \
  -p 8000:8000 \
  --cpuset-cpus 0-55 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --restart always \
  --ipc=host \
  vllm/vllm-openai:v0.8.5 \
  --model /model/Qwen3-32B-AWQ \
  --served-model-name coder \
  --tensor-parallel-size 4 \
  --dtype half \
  --quantization awq \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.93 \
  --block-size 32 \
  --enable-chunked-prefill \
  --swap-space 16 \
  --tokenizer-pool-size 56 \
  --disable-custom-all-reduce \
  --chat-template /app/qwen3_nonthinking.jinja
```

### Command Breakdown:

- `docker run --rm -it --gpus all -p 8000:8000`: Standard Docker flags to run interactively, allocate all GPUs, and map port 8000.
- `-v /home/llm/model/qwen/Qwen3-32B-AWQ:/models/Qwen3-32B-AWQ`: Mounts your local model directory (`/home/llm/model/qwen/Qwen3-32B-AWQ`) into the container at `/models/Qwen3-32B-AWQ`.
- `-v /home/llm/model/qwen/Qwen3-32B-AWQ/qwen3_nonthinking.jinja:/app/qwen3_nonthinking.jinja`: Mounts the custom chat template into the container. This template ensures `enable_thinking=False` behavior.
- `vllm/vllm-openai:v0.8.5`: Specifies the required Docker image and version.
- `--model /models/Qwen3-32B-AWQ`: Specifies the path to the model inside the container. This points to the locally mounted model files.
- `--chat-template /app/qwen3_nonthinking.jinja`: Instructs vLLM to use our custom template to disable thinking mode completely.
- `--tensor-parallel-size 4`: Utilizes 4 GPUs for inference.
- `--max-model-len 32768`: Sets the maximum context length to 32,768 tokens.
- `--quantization awq`: Specifies that the model is AWQ quantized.
- `--dtype auto`: Allows vLLM to automatically determine the appropriate data type (usually `half` or `bfloat16` for AWQ models).
- `--host 0.0.0.0 --port 8000`: Configures the vLLM server to listen on all network interfaces within the container on port 8000.

## OpenAI Compatible API

Once the server is running, it will provide an OpenAI-compatible API endpoint at `http://localhost:8000/v1`.

## Notes

- This setup ensures that the `enable_thinking` parameter is effectively `False` at the server level, meaning the model will not produce `<think>...</think>` blocks in its responses.
- According to Qwen documentation for vLLM 0.8.5, when using a chat template to disable thinking (or passing `enable_thinking=False` via API calls), the `--enable-reasoning-parser` and `--reasoning-parser` flags should not be used as they are incompatible.
