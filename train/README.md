## Planner fine-tuning

For fine-tuning a Mistral-7B-v0.1 based planner on trajectories generated in the GPT-4 based [agent simulation](../simulation/README.md), first create and activate the `bot-with-plan-autotrain` conda environment

```shell
conda env create -f environment-autotrain.yml
conda activate bot-with-plan-autotrain
```

and then run the following command to start QLoRA fine-tuning:

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

During fine-tuning, the planner learns to select from the set of tools available in the trajectories and doesn't need to be configured with tools at inference time. This significantly reduces planner prompt sizes and inference latencies. A fine-tuned QLoRA model is available in the [krasserm/gba-planner-7B-v0.1](https://huggingface.co/krasserm/gba-planner-7B-v0.1) repository.

Switch back to the `bot-with-plan` conda environment and optionally inspect a few fine-tuned planner outputs by comparing them to GPT-4 based planner outputs.

```shell
python train/planner/validate.py \
  --model_dir gba-planner-7B \
  --dataset_dir output/dataset
```

 Merge the trained QLoRA model back into the base model.

```shell
python train/planner/merge.py \
  --model_dir gba-planner-7B \
  --output_dir gba-planner-7B-merged
```

## GGUF conversion and quantization

Convert the fine-tuned planner model into a llama.cpp compatible format and quantize it. Quantized models are also available in the [krasserm/gba-planner-7B-v0.1-GGUF](https://huggingface.co/krasserm/gba-planner-7B-v0.1-GGUF) repo and can be served with a llama.cpp server.

The following commands require a local copy of the llama.cpp repository (built with CUDA support). **TODO**: show how to do this with a llama.cpp Docker container. In the root directory of the llama.cpp repository run:

```shell
ln -s /path/to/bot-with-plan gba

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
