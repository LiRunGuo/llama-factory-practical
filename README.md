## LLaMA-Factory 微调实战（项目复刻）

本项目依据 `llama-factory微调.txt` 内容复刻与整理，提供从环境检查、安装、WebUI 启动、数据准备与训练命令，到评测、导出合并、GGUF 转换与 Ollama 部署、API 服务调用的完整示例与脚本。

---

### 目录结构

```
llama-factory实战项目/
  ├─ README.md
  ├─ requirements.txt
  ├─ env.example
  ├─ examples/
  │   ├─ train_lora_qwen_windows.yaml
  │   └─ eval_lora_llama3.yaml
  └─ src/
      ├─ gpu_check.py
      ├─ download_model_modelscope.py
      ├─ inference_transformers.py
      ├─ llama_factory_api_client.py
      └─ ollama_api_examples.py
```

---

### 环境准备

硬件/驱动：
- 更新 NVIDIA 驱动，安装 CUDA（建议 12.1 或更高），并安装 PyTorch（匹配 CUDA 版本）
- 检查：`nvidia-smi` 与 `nvcc -V`

Python 与依赖：
```
conda create -n llama_factory python=3.10
conda activate llama_factory
pip install -r requirements.txt

# 安装 LLaMA-Factory（建议在其仓库根目录执行）
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

量化（Windows QLoRA 可选）：
```
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.2.post2-py3-none-win_amd64.whl
pip install autoawq  # 如需 AWQ 量化模型支持
```

GPU 校验脚本（本项目已提供）：
```
python src/gpu_check.py
```

---

### WebUI 启动

确保已完成 LLaMA-Factory 安装（见上）。在激活的虚拟环境执行：
```
llamafactory-cli train -h
llamafactory-cli webui
# 或：CUDA_VISIBLE_DEVICES=0 llamafactory-cli webui

# 如需开启 Gradio share 或修改端口：
CUDA_VISIBLE_DEVICES=0 GRADIO_SHARE=1 GRADIO_SERVER_PORT=7860 llamafactory-cli webui
```

> 注意：WebUI 当前支持单机单卡和单机多卡，多机多卡请使用命令行。

---

### 模型下载

ModelScope 方式（脚本已提供）：
```
python src/download_model_modelscope.py
```

Transformers 推理测试（脚本已提供）：
```
python src/inference_transformers.py
```

---

### 数据与任务

- SFT 常用 Alpaca 格式；DPO 偏好数据常用 ShareGPT 格式。
- 示例数据可参考：`hiyouga/DPO-En-Zh-20k`，`nvidia/HelpSteer2`；自定义数据示例见文档引用。

---

### 训练（命令行）

Windows 一行命令示例（Qwen2-7B-AWQ + LoRA + bitsandbytes 4bit）：
```
llamafactory-cli train --stage sft --do_train True --model_name_or_path Qwen/Qwen2-7B-Instruct-AWQ --preprocessing_num_workers 16 --finetuning_type lora --template qwen --flash_attn auto --dataset_dir data --dataset huanhuan_chat --cutoff_len 1024 --learning_rate 5e-05 --num_train_epochs 3.0 --max_samples 100000 --per_device_train_batch_size 2 --gradient_accumulation_steps 8 --lr_scheduler_type cosine --max_grad_norm 1.0 --logging_steps 5 --save_steps 100 --warmup_steps 0 --optim adamw_torch --packing False --report_to none --output_dir saves\Qwen2-7B-int4-Chat\lora\train_YYYY-MM-DD-HH-MM-SS --fp16 True --plot_loss True --ddp_timeout 180000000 --include_num_input_tokens_seen True --quantization_bit 4 --quantization_method bitsandbytes --lora_rank 8 --lora_alpha 16 --lora_dropout 0 --lora_target all
```

YAML 配置示例（见 `examples/train_lora_qwen_windows.yaml`）：
```
llamafactory-cli train examples/train_lora_qwen_windows.yaml
```

断点续训：
```
--resume_from_checkpoint <checkpoint_dir> --output_dir <new_dir>
```

---

### 评测

命令行评测（MMLU 示例）：
```
llamafactory-cli eval examples/eval_lora_llama3.yaml
```

Windows 示例（中文 CMMMU）：
```
llamafactory-cli eval --model_name_or_path Qwen/Qwen2-7B-Instruct-AWQ --adapter_name_or_path F:\sotaAI\LLaMA-Factory\saves\Qwen2-7B-int4-Chat\lora\train_xxx --finetuning_type lora --template qwen --task cmmlu_test --lang zh --n_shot 3 --batch_size 1
```

批量推理（生成评分数据示例，来自文档）请参考 README 末尾“参考命令”段落。

---

### 导出与合并（LoRA → 全量权重）

合并导出：
```
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
  --model_name_or_path <base_model_dir> \
  --adapter_name_or_path <lora_dir> \
  --template llama3 \
  --finetuning_type lora \
  --export_dir merged-model-path \
  --export_size 2 \
  --export_device cpu \
  --export_legacy_format False
```

GGUF 转换（llama.cpp）：
```
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install --editable .
python convert_hf_to_gguf.py <merged-model-path>
```

---

### Ollama 部署

安装与运行（Windows 原生可用）：
- 下载：`https://ollama.com/download`
- 默认模型存储在 `C:\Users\<User>\.ollama`，可通过环境变量重定向：
  - `OLLAMA_MODELS` 指定模型目录
  - `OLLAMA_BASE_URL` 指定 API 地址（默认 `http://127.0.0.1:11434`）

自定义 Modelfile（GGUF 路径替换为你的导出模型）：
```
FROM C:/path/to/your/merged-model.gguf
PARAMETER temperature 1
SYSTEM """
You are Mario from Super Mario Bros. Answer as Mario, the assistant, only.
"""
```
注册与运行：
```
ollama create huanhuan -f path/to/Modelfile
ollama run huanhuan
```

Ollama API（脚本已提供）：
```
python src/ollama_api_examples.py
```

---

### API 服务

LLaMA-Factory OpenAI 兼容 API：
```
CUDA_VISIBLE_DEVICES=0 API_PORT=8000 llamafactory-cli api \
  --model_name_or_path <base_or_merged_model> \
  --adapter_name_or_path <optional_lora_dir> \
  --template llama3 \
  --finetuning_type lora

python src/llama_factory_api_client.py  # 客户端示例
```

可选：vLLM 推理后端（Linux 支持）：
```
CUDA_VISIBLE_DEVICES=0 API_PORT=8000 llamafactory-cli api \
  --model_name_or_path <merged-model-path> \
  --template llama3 \
  --infer_backend vllm \
  --vllm_enforce_eager
```

---

### 参考命令（批量推理）

```
llamafactory-cli train \
  --stage sft \
  --do_predict \
  --model_name_or_path <base_model_dir> \
  --adapter_name_or_path <lora_dir> \
  --eval_dataset alpaca_gpt4_zh,identity,adgen_local \
  --dataset_dir ./data \
  --template llama3 \
  --finetuning_type lora \
  --output_dir ./saves/lora/predict \
  --overwrite_cache \
  --overwrite_output_dir \
  --cutoff_len 1024 \
  --preprocessing_num_workers 16 \
  --per_device_eval_batch_size 1 \
  --max_samples 20 \
  --predict_with_generate True \
  --max_new_tokens 512 \
  --top_p 0.7 \
  --temperature 0.95
```

---

### 常见问题

- 识别不到 GPU：优先检查驱动/CUDA 与 PyTorch 版本匹配，运行 `src/gpu_check.py` 自检。
- bitsandbytes（Windows）：优先使用文档指定的第三方 wheel；或在 Linux/WSL2 环境运行。
- 输出目录/检查点：Windows 使用反斜杠 `\`，注意转义或使用原始字符串。


