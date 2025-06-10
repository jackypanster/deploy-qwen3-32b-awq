## 需求

### 一、模型部署与配置

1.  使用 vLLM 本地部署 `Qwen3-32B-AWQ` 模型：[https://www.modelscope.cn/models/Qwen/Qwen3-32B-AWQ/summary](https://www.modelscope.cn/models/Qwen/Qwen3-32B-AWQ/summary)
2.  实现 32,768 个 tokens 的上下文长度和 OpenAI 兼容 API，适用于生产环境。
3.  通过精细调整部署参数，在指定的 GPU 资源下最大化模型性能。
4.  模型文件已经下载到本地：`/home/llm/model/qwen/Qwen3-32B-AWQ`
5.  设置：`enable_thinking=False`
    *   这种模式在需要通过禁用思考来提高效率的场景中特别有用。
    *   示例代码：
        ```python
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Setting enable_thinking=False disables thinking mode
        )
        ```
    *   在这种模式下，模型不会生成任何思考内容，也不会包含 `<think>...</think>` 块。
6.  可以参考以下文章：[https://jackypanster.github.io/ai-stream/posts/deploy-qwen3/](https://jackypanster.github.io/ai-stream/posts/deploy-qwen3/)

### 二、系统要求 🖥️

#### 硬件配置

*   4 块 NVIDIA GPU (每块 22GB 显存，总计 88GB)
*   512GB 系统内存
*   2TB SSD 存储
*   56 核 CPU

#### 软件环境

*   Ubuntu 24.04
*   NVIDIA-SMI `570.153.02` (Driver Version: `570.153.02`, CUDA Version: `12.8`)
*   Docker Image: `vllm/vllm-openai:v0.8.5`
