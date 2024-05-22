## Planner fine-tuning

For fine-tuning the planner module, trajectories from a GPT-4 based [agent simulation](../simulation/README.md) are
used. To start fine-tuning on these trajectories run the following command in the `grammar-based-agents-autotrain`
conda environment:

```shell
autotrain llm \
  --project-name gba-planner-7B \
  --train \
  --model "mistralai/Mistral-7B-v0.1" \
  --data-path output/dataset \
  --train-split train \
  --valid-split validation \
  --text_column text \
  --lr 0.0002 \
  --epochs 3 \
  --train-batch-size 1 \
  --warmup_ratio 0.03 \
  --gradient-accumulation 2 \
  --optimizer adamw_torch \
  --scheduler linear \
  --weight_decay 0 \
  --seed 0 \
  --use-peft \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --logging_steps 10 \
  --save_total_limit 1 \
  --mixed-precision fp16 \
  --quantization int4 \
  --block_size 1024 \
  --model_max_length 1024
```

and then evaluate the model back in the `grammar-based-agents` conda environment:

```shell
python train/planner/validate.py \
  --model_dir gba-planner-7B \
  --dataset_dir output/dataset
```

Then merge the trained LoRA adapter back into the base model.

```shell
python train/planner/merge.py \
  --model_dir gba-planner-7B \
  --output_dir gba-planner-7B-merged
```

## Planner model conversion and quantization

Convert the fine-tuned planner model into a llama.cpp compatible format and quantize it to 8 bit. This requires a local
copy of the llama.cpp repository (built with CUDA support). **TODO**: show how to do this with a llama.cpp Docker container.

In the root directory of the llama.cpp repository run:

```shell
ln -s /path/to/grammar-based-agents gba

python convert.py gba/gba-planner-7B-merged \
  --outfile gba/gba-planner-7B-v0.1.gguf \
  --outtype f16

./build/bin/quantize \
  gba/gba-planner-7B-v0.1.gguf \
  gba/gba-planner-7B-v0.1-Q8_0.gguf Q8_0

./build/bin/quantize \
  gba/gba-planner-7B-v0.1.gguf \
  gba/gba-planner-7B-v0.1-Q4_K_M.gguf Q4_K_M
```

The quantized models can be downloaded from the [krasserm/gba-planner-7B-v0.1-GGUF](https://huggingface.co/krasserm/gba-planner-7B-v0.1-GGUF)
repo and served with a llama.cpp server.
