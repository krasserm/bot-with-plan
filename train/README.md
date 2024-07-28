## Planner fine-tuning

Planner fine-tuning uses a Mistral-7B base model and trajectories generated in a GPT-4 based [agent simulation](../simulation/README.md). During fine-tuning, a planner learns to describe the task for the next step and to select an appropriate tool that executes one or more actions derived from the task description. The set of available tools is learned from the trajectories. There's no need to prompt the planner with available tools at inference time which significantly reduces prompt sizes and inference latencies.

### gba-planner-7B-v0.1

For fine-tuning a Mistral-7B-v0.1 based planner, first create and activate the `bot-with-plan-autotrain` conda environment

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

mv gba-planner-7B gba-planner-7B-v0.1
```

 The loss is computed over the full sequence (prompt not masked).

The fine-tuned QLoRA model is available in the [krasserm/gba-planner-7B-v0.1](https://huggingface.co/krasserm/gba-planner-7B-v0.1) repository. Quantized GGUF versions are available in the [krasserm/gba-planner-7B-v0.1-GGUF](https://huggingface.co/krasserm/gba-planner-7B-v0.1-GGUF) repository.

### gba-planner-7B-v0.2

Version 0.2 planner models are based on Mistral-7B-v0.3 and trained with different loss functions:

- `gba-planner-7B-v0.2` is fine-tuned with a loss over the full sequence i.e. prompt and completion tokens
- `gba-planner-7B-completion-only-v0.2` is fine-tuned with a loss over completion tokens only (prompt masked)

The following commands have been tested on a machine with 4 RTX 3080Ti GPUs (12GB VRAM each). Fine-tuning is done with the custom [sft_qlora.py](planner/sft_qlora.py) script instead of `autotrain` as `autotrain` doesn't support completion-only fine-tuning (at the time of writing). In conda environment `bot-with-plan` run:

```shell
accelerate launch \
  --config_file train/planner/sft_qlora.yaml train/planner/sft_qlora.py \
  --completion_only=false \
  --packing=false \
  --num_epochs=2 \
  --gradient_accumulation_steps=2 \
  --output_dir=gba-planner-7B-v0.2

accelerate launch \
  --config_file train/planner/sft_qlora.yaml train/planner/sft_qlora.py \
  --completion_only=true \
  --packing=false \
  --num_epochs=2 \
  --gradient_accumulation_steps=2 \
  --output_dir=gba-planner-7B-completion-only-v0.2
```

Training the model with FlashAttention-2 requires `packing=true` and computing the loss over the full sequence:

```shell
accelerate launch \
  --config_file train/planner/sft_qlora.yaml train/planner/sft_qlora.py \
  --completion_only=false \
  --packing=true \
  --flash_attn=true \
  --num_epochs=2 \
  --gradient_accumulation_steps=2 \
  --output_dir=gba-planner-7B-v0.2
```

These commands replicate the model across GPUs with DDP. For distributed FSDP training (experimental), which allows larger batch sizes without gradient accumulation, use the [sft_qlora_fsdp.py](planner/sft_qlora_fsdp.py) script:

```shell
accelerate launch \
  --config_file train/planner/sft_qlora_fsdp.yaml train/planner/sft_qlora_fsdp.py \
  --completion_only=false \
  --packing=false \
  --num_epochs=2 \
  --per_device_batch_size=4 \
  --output_dir=gba-planner-7B-v0.2

accelerate launch \
  --config_file train/planner/sft_qlora_fsdp.yaml train/planner/sft_qlora_fsdp.py \
  --completion_only=true \
  --packing=false \
  --num_epochs=2 \
  --per_device_batch_size=4 \
  --output_dir=gba-planner-7B-completion-only-v0.2
```

Fine-tuned models are available in the [krasserm/gba-planner-v0.2](https://huggingface.co/krasserm/gba-planner-7B-v0.2) and [krasserm/gba-planner-v0.2-completion-only](https://huggingface.co/krasserm/gba-planner-7B-completion-only-v0.2) repositories.

 After fine-tuning, optionally inspect a few model outputs generated from validation set prompts and compare them to GPT-4 based planner outputs:

```shell
python train/planner/validate.py \
  --model_dir gba-planner-7B-v0.2 \
  --dataset_dir output/dataset

python train/planner/validate.py \
  --model_dir gba-planner-7B-completion-only-v0.2 \
  --dataset_dir output/dataset
```

With FlashAttention-2:

```shell
python train/planner/validate.py \
  --model_dir gba-planner-7B-v0.2 \
  --dataset_dir output/dataset \
  --flash_attn=true
```

 Merge the trained QLoRA model back into the base model:

```shell
python train/planner/merge.py \
  --model_dir gba-planner-7B-v0.2 \
  --output_dir gba-planner-7B-v0.2-merged

python train/planner/merge.py \
  --model_dir gba-planner-7B-completion-only-v0.2 \
  --output_dir gba-planner-7B-completion-only-v0.2-merged
```

Convert them to GGUF format

```shell
docker run --gpus all --rm -v $(realpath .):/project ghcr.io/ggerganov/llama.cpp:full-cuda--b1-17b291a --convert \
  /project/gba-planner-7B-v0.2-merged \
  --outfile /project/models/gba-planner-7B-v0.2.gguf \
  --outtype bf16

docker run --gpus all --rm -v $(realpath .):/project ghcr.io/ggerganov/llama.cpp:full-cuda--b1-17b291a --convert \
  /project/gba-planner-7B-completion-only-v0.2-merged \
  --outfile /project/models/gba-planner-7B-completion-only-v0.2.gguf \
  --outtype bf16
```

 and quantize them:

```shell
docker run --gpus all --rm -v $(realpath .):/project ghcr.io/ggerganov/llama.cpp:full-cuda--b1-17b291a --quantize \
  /project/models/gba-planner-7B-v0.2.gguf \
  /project/models/gba-planner-7B-v0.2-Q8_0.gguf Q8_0

docker run --gpus all --rm -v $(realpath .):/project ghcr.io/ggerganov/llama.cpp:full-cuda--b1-17b291a --quantize \
  /project/models/gba-planner-7B-completion-only-v0.2.gguf \
  /project/models/gba-planner-7B-completion-only-v0.2-Q8_0.gguf Q8_0
```

Quantized models are available in the [krasserm/gba-planner-7B-v0.2-GGUF](https://huggingface.co/krasserm/gba-planner-7B-v0.2-GGUF) and [krasserm/gba-planner-7B-completion-only-v0.2-GGUF](https://huggingface.co/krasserm/gba-planner-7B-completion-only-v0.2-GGUF) repositories.
