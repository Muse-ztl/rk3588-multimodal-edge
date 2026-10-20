# 把 Qwen2.5-1.5B 量化成 w4a16（rkllm-toolkit 1.2.3）

仓库里跑的演示模型是 `Qwen2.5-1.5B.rkllm`，运行时日志显示 `model_dtype: W8A8`。
如果目标是 w4a16（权重 4bit、激活 16bit，进一步压带宽），按下面步骤重新导出。

> 注意：w4a16 对 1.5B 这种小模型收益主要在**内存带宽和模型体积**（约 2.0GB → 1.2GB），
> 端侧 NPU 的算力并不是瓶颈；量化后建议用同一批 Prompt 对比输出质量再上线。

## 1. 准备环境（x86 Linux，Python 3.10）

rkllm-toolkit 对 transformers 版本敏感，单独建环境，别和 rknn-toolkit2 混装：

```bash
python3 -m venv ~/venv_rkllm && source ~/venv_rkllm/bin/activate
pip install rkllm-toolkit==1.2.3
pip install "transformers==4.55.2" "safetensors==0.5.3" torch --index-url https://download.pytorch.org/whl/cpu
```

## 2. 准备基座模型

```bash
# ModelScope（国内直连）
pip install modelscope
python - <<'PY'
from modelscope import snapshot_download
print(snapshot_download("Qwen/Qwen2.5-1.5B-Instruct", local_dir="/data/Qwen2.5-1.5B-Instruct"))
PY
```

## 3. 生成校准数据

`rkllm-toolkit` 自带脚本，从基座模型的 tokenizer 采样即可：

```bash
cd $(python -c "import rkllm, os; print(os.path.dirname(rkllm.__file__))")/../..
python examples/rkllm_api_demo/export/generate_data_quant.py \
    -m /data/Qwen2.5-1.5B-Instruct \
    -o ./quant_calib_data.json
```

## 4. 导出 w4a16

```python
from rkllm.api import RKLLM

llm = RKLLM()
llm.load_huggingface(model="/data/Qwen2.5-1.5B-Instruct", model_lora=None, device="cpu")
llm.build(
    do_quantization=True,
    optimization_level=1,
    quantized_dtype="w4a16",          # 关键：w4a16 / w8a8 / w4a16_g128 等
    quantized_algorithm="normal",
    target_platform="rk3588",
    num_npu_core=3,
    dataset="./quant_39k.json",       # 上一步生成的校准数据
    hybrid_rate=0.0,                  # 需要精度补偿时可调到 0.05~0.1（部分层保留 fp16）
)
llm.export_rkllm("./Qwen2.5-1.5B-w4a16.rkllm")
```

## 5. 板上验证

```bash
~/mm_env/bin/python scripts/smoke_llm.py ./Qwen2.5-1.5B-w4a16.rkllm
```

跑通标准：`rkllm_init` 日志里 `model_dtype` 显示 `W4A16`，且同一 Prompt 的输出与
w8a8 版本语义一致（本仓库的 Prompt 只有一句话，差异通常只体现在个别措辞上）。
