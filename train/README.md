## Planner fine-tuning

For fine-tuning the planner module, trajectories from a GPT-4 based [agent simulation](../simulation/README.md) are
used. To start fine-tuning on these trajectories run the following command in the `grammar-based-agents-autotrain` 
conda environment:

```shell  
autotrain llm \
  --project-name project-1 \
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
  --checkpoint_dir project-1/checkpoint-999 \
  --dataset_dir output/dataset
```

Depending on the number of actual training steps you may need to adjust the checkpoint number. Finally, merge the 
trained LoRA adapter back into the base model.
    
```shell
python train/planner/merge.py \
  --checkpoint_dir project-1/checkpoint-999 \
  --output_dir project-1/checkpoint-999_merged
```

## Planner model conversion and quantization

Convert the fine-tuned planner model to a llama.cpp compatible format and quantizes is to 8 bit. This requires a local 
copy of the llama.cpp repository (built with CUDA support). In the root directory of the llama.cpp repository run: 

```shell
# TODO: remove hard-coded paths

python convert.py /home/martin/Development/krasserm/grammar-based-agents/project-1/checkpoint-999_merged \
  --outfile /home/martin/Development/krasserm/grammar-based-agents/project-1/checkpoint-999_merged.gguf \
  --outtype f16

./build/bin/quantize \
  /home/martin/Development/krasserm/grammar-based-agents/project-1/checkpoint-999_merged.gguf \
  /home/martin/Development/krasserm/grammar-based-agents/project-1/checkpoint-999_merged-Q8_0.gguf Q8_0
```

The quantized model can be downloaded from [this model repo](https://huggingface.co/krasserm/checkpoint-999) and served 
with a llama.cpp server.

