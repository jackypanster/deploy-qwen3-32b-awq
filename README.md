[English](#english) | [中文](#chinese)

<a name="english"></a>
## English

# Deploying Qwen3-32B-AWQ with vLLM

This document outlines the steps to deploy the `Qwen/Qwen3-32B-AWQ` model using vLLM, adhering to the requirements specified in `prd.md`.

## Prerequisites

- Docker installed.
- NVIDIA drivers installed on the host machine (as per `prd.md`: Driver Version 570.153.02, CUDA 12.8 compatible).
- Access to 4 NVIDIA GPUs with at least 22GB VRAM each.
- The `qwen3_nonthinking.jinja` chat template file in this directory (project root).

## Deployment Command

To start the vLLM server with the OpenAI-compatible API, run the following command from the root of this workspace (`/home/user/workspace/deploy-qwen3-32b-awq`):

```bash
docker run -d \
  --runtime=nvidia \
  --gpus=all \
  --name coder \
  -v /home/llm/model/qwen/Qwen3-32B-AWQ:/model/Qwen3-32B-AWQ \
  -v $(pwd)/qwen3_nonthinking.jinja:/app/qwen3_nonthinking.jinja \
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

- `-d`: Runs the container in detached mode (in the background).
- `--runtime=nvidia`: Specifies the NVIDIA container runtime.
- `--gpus=all`: Allocates all available NVIDIA GPUs to the container.
- `--name coder`: Assigns the name "coder" to the container for easy reference.
- `-v /home/llm/model/qwen/Qwen3-32B-AWQ:/model/Qwen3-32B-AWQ`: Mounts the local model directory from the host (`/home/llm/model/qwen/Qwen3-32B-AWQ`) to `/model/Qwen3-32B-AWQ` inside the container.
- `-v $(pwd)/qwen3_nonthinking.jinja:/app/qwen3_nonthinking.jinja`: Mounts the custom `qwen3_nonthinking.jinja` chat template from the current working directory (project root) on the host to `/app/qwen3_nonthinking.jinja` inside the container.
- `-p 8000:8000`: Maps port 8000 on the host to port 8000 in the container.
- `--cpuset-cpus 0-55`: Restricts the container to use CPU cores 0 through 55.
- `--ulimit memlock=-1`: Sets an unlimited memory lock limit.
- `--ulimit stack=67108864`: Sets the stack size limit to 64MB.
- `--restart always`: Configures the container to restart automatically if it stops.
- `--ipc=host`: Uses the host system's IPC namespace, which can improve performance for multi-GPU communication.
- `vllm/vllm-openai:v0.8.5`: Specifies the Docker image to use.
- `--model /model/Qwen3-32B-AWQ`: Tells vLLM where to find the model files inside the container.
- `--served-model-name coder`: Sets the name of the model as it will be served by the API.
- `--tensor-parallel-size 4`: Configures vLLM to use 4 GPUs for tensor parallelism.
- `--dtype half`: Sets the data type for model weights and activations to half-precision floating point.
- `--quantization awq`: Specifies that the model uses AWQ (Activation-aware Weight Quantization).
- `--max-model-len 32768`: Sets the maximum sequence length the model can handle to 32,768 tokens.
- `--max-num-batched-tokens 4096`: Sets the maximum number of tokens in a batch.
- `--gpu-memory-utilization 0.93`: Instructs vLLM to use up to 93% of GPU memory.
- `--block-size 32`: Sets the block size for the paged attention KV cache.
- `--enable-chunked-prefill`: Enables chunked prefilling for long sequences.
- `--swap-space 16`: Allocates 16GB of CPU RAM for swapping GPU memory (PagedAttention).
- `--tokenizer-pool-size 56`: Sets the size of the tokenizer pool.
- `--disable-custom-all-reduce`: Disables vLLM's custom all-reduce kernel (recommended for >2 PCIe-only GPUs).
- `--chat-template /app/qwen3_nonthinking.jinja`: Specifies the path to the custom chat template file inside the container.

## OpenAI Compatible API

Once the server is running, it will provide an OpenAI-compatible API endpoint at `http://localhost:8000/v1` (or your server's IP address if not running on localhost).

## Notes

- This setup ensures that the `enable_thinking` parameter is effectively `False` at the server level, meaning the model will not produce `<think>...</think>` blocks in its responses.
- According to Qwen documentation for vLLM 0.8.5, when using a chat template to disable thinking (or passing `enable_thinking=False` via API calls), the `--enable-reasoning-parser` and `--reasoning-parser` flags should not be used as they are incompatible.

---

<a name="chinese"></a>
## 中文 (Chinese)

# 使用 vLLM 部署 Qwen3-32B-AWQ

本文档概述了遵循 `prd.md` 中指定的要求，使用 vLLM 部署 `Qwen/Qwen3-32B-AWQ` 模型的步骤。

## 先决条件

- 已安装 Docker。
- 主机上已安装 NVIDIA 驱动程序 (根据 `prd.md`：驱动程序版本 570.153.02，与 CUDA 12.8 兼容)。
- 可访问 4 个 NVIDIA GPU，每个 GPU 至少有 22GB VRAM。
- 此目录（项目根目录）中包含 `qwen3_nonthinking.jinja` 聊天模板文件。

## 部署命令

要启动具有 OpenAI 兼容 API 的 vLLM 服务器，请从此工作区的根目录 (`/home/user/workspace/deploy-qwen3-32b-awq`) 运行以下命令：

```bash
docker run -d \
  --runtime=nvidia \
  --gpus=all \
  --name coder \
  -v /home/llm/model/qwen/Qwen3-32B-AWQ:/model/Qwen3-32B-AWQ \
  -v $(pwd)/qwen3_nonthinking.jinja:/app/qwen3_nonthinking.jinja \
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

### 命令详解:

- `-d`: 以分离模式（后台）运行容器。
- `--runtime=nvidia`: 指定 NVIDIA 容器运行时。
- `--gpus=all`: 将所有可用的 NVIDIA GPU 分配给容器。
- `--name coder`: 为容器分配名称 "coder"，方便引用。
- `-v /home/llm/model/qwen/Qwen3-32B-AWQ:/model/Qwen3-32B-AWQ`: 将宿主机的本地模型目录 (`/home/llm/model/qwen/Qwen3-32B-AWQ`) 挂载到容器内的 `/model/Qwen3-32B-AWQ`。
- `-v $(pwd)/qwen3_nonthinking.jinja:/app/qwen3_nonthinking.jinja`: 将宿主机当前工作目录（项目根目录）下的自定义 `qwen3_nonthinking.jinja` 聊天模板挂载到容器内的 `/app/qwen3_nonthinking.jinja`。
- `-p 8000:8000`: 将宿主机的 8000 端口映射到容器的 8000 端口。
- `--cpuset-cpus 0-55`: 限制容器使用 CPU 核心 0 到 55。
- `--ulimit memlock=-1`: 设置无限制的内存锁定。
- `--ulimit stack=67108864`: 设置堆栈大小限制为 64MB。
- `--restart always`: 配置容器在停止时自动重启。
- `--ipc=host`: 使用宿主系统的 IPC 命名空间，可提高多 GPU 通信性能。
- `vllm/vllm-openai:v0.8.5`: 指定要使用的 Docker 镜像。
- `--model /model/Qwen3-32B-AWQ`: 告知 vLLM 在容器内何处查找模型文件。
- `--served-model-name coder`: 设置 API 服务时使用的模型名称。
- `--tensor-parallel-size 4`: 配置 vLLM 使用 4 个 GPU 进行张量并行。
- `--dtype half`: 将模型权重和激活值的数据类型设置为半精度浮点数。
- `--quantization awq`: 指定模型使用 AWQ（激活感知权重化）。
- `--max-model-len 32768`: 设置模型可以处理的最大序列长度为 32768 个令牌。
- `--max-num-batched-tokens 4096`: 设置批处理中的最大令牌数。
- `--gpu-memory-utilization 0.93`: 指示 vLLM 使用高达 93% 的 GPU 显存。
- `--block-size 32`: 设置 PagedAttention KV 缓存的块大小。
- `--enable-chunked-prefill`: 为长序列启用分块预填充。
- `--swap-space 16`: 分配 16GB CPU RAM 用于交换 GPU 显存 (PagedAttention)。
- `--tokenizer-pool-size 56`: 设置分词器池的大小。
- `--disable-custom-all-reduce`: 禁用 vLLM 的自定义 all-reduce 内核（推荐用于多于2个仅PCIe连接的GPU）。
- `--chat-template /app/qwen3_nonthinking.jinja`: 指定容器内自定义聊天模板文件的路径。

## OpenAI 兼容 API

服务器运行后，它将在 `http://localhost:8000/v1`（如果不在本地主机上运行，则为服务器的 IP 地址）提供一个 OpenAI 兼容的 API 端点。

## 注意事项

- 此设置确保 `enable_thinking` 参数在服务器级别有效为 `False`，这意味着模型在其响应中不会产生 `<think>...</think>` 块。
- 根据 Qwen vLLM 0.8.5 的文档，当使用聊天模板禁用思考（或通过 API 调用传递 `enable_thinking=False`）时，不应使用 `--enable-reasoning-parser` 和 `--reasoning-parser` 标志，因为它们不兼容。
