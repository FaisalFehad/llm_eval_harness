#!/usr/bin/env python3
"""
V14 LoRA training — Unsloth + TRL on Lambda GH200.

Speed optimisations over standard HuggingFace:
  - Unsloth FastLanguageModel  (2-5x faster LoRA kernels, custom Triton)
  - Unsloth gradient_checkpointing  (checkpoints every 4th layer vs every layer)
  - Flash Attention 2          (O(n) memory at seq_len=6144)
  - Fused AdamW                (kernel-level optimizer step)
  - completion_only_loss=True  (train on assistant response only)
  - Liger Kernel fallback      (if unsloth not available, eliminates 40GB logit tensor)

Falls back to standard HuggingFace PEFT if unsloth is not installed.

Usage (Lambda):
    python3 finetune/train_v14.py

Resume:
    python3 finetune/train_v14.py --resume-from finetune/adapters_v14/checkpoint-1000

Checkpoints: finetune/adapters_v14/checkpoint-{step}/
"""

import argparse
import sys
from pathlib import Path

import torch

# Unsloth MUST be imported before trl/transformers/peft to patch their kernels.
# Falls back gracefully if not installed.
try:
    from unsloth import FastLanguageModel
    UNSLOTH = True
except ImportError:
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    UNSLOTH = False

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# ── LoRA target modules for Qwen2.5 / Qwen3 ──────────────────────────────────
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def load_model_unsloth(args):
    """Load model + LoRA via Unsloth (2-5x faster kernels)."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_len,
        dtype=torch.bfloat16,
        load_in_4bit=args.load_4bit,  # 4-bit saves ~6GB — needed for VL models (Qwen3.5)
        trust_remote_code=True,
    )
    # lora_dropout MUST be 0 for unsloth to patch LoRA matrices with fast kernels.
    # With dropout > 0, unsloth falls back to slow HF kernels for LoRA layers.
    # Regularisation is still provided by weight_decay=0.01 + LoRA rank constraint.
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        target_modules=LORA_TARGETS,
        bias="none",
        # "unsloth" mode: checkpoints every 4th layer (vs every layer in HF)
        # ~2x faster backward pass at cost of ~30% more VRAM vs full grad_ckpt
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


def load_model_hf(args):
    """Fallback: standard HuggingFace PEFT + liger_kernel."""
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(
        description="V14 LoRA training — Unsloth + TRL on Lambda GPU"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--data-dir", default="data/v14")
    parser.add_argument("--output-dir", default="finetune/adapters_v14")

    # LoRA — identical to V13.1
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Training
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Per-device batch (32 for unsloth smart grad_ckpt; 48 for HF fallback)")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=6144)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--dataloader-workers", type=int, default=8)
    parser.add_argument("--load-4bit", action="store_true",
                        help="Load base model in 4-bit (saves ~6GB VRAM — needed for Qwen3.5 VL models)")
    args = parser.parse_args()

    effective_batch = args.batch_size * args.grad_accum
    backend = "Unsloth" if UNSLOTH else "HuggingFace PEFT + liger_kernel"
    print(f"V14 training — {args.model}")
    print(f"  Backend:         {backend}")
    print(f"  Data:            {args.data_dir}")
    print(f"  Output:          {args.output_dir}")
    dropout_actual = 0 if UNSLOTH else args.lora_dropout
    print(f"  LoRA:            rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={dropout_actual}" +
          (" (forced 0 — unsloth requires this for full kernel patching)" if UNSLOTH else ""))
    print(f"  Steps:           {args.max_steps} (save every {args.save_every}, eval every {args.eval_every})")
    print(f"  Effective batch: {effective_batch} (per_device={args.batch_size} × grad_accum={args.grad_accum})")
    print(f"  LR:              {args.lr} (warmup={args.warmup_steps} steps, linear decay)")
    print(f"  Max seq len:     {args.max_seq_len}")
    print(f"  Base precision:  {'4-bit (QLoRA)' if args.load_4bit else 'bfloat16 (full)'}")
    print()

    # ── Verify data ────────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    for fname in ("train.jsonl", "valid.jsonl"):
        if not (data_dir / fname).exists():
            print(f"ERROR: {data_dir / fname} not found.")
            sys.exit(1)

    # ── Load model + tokenizer ─────────────────────────────────────────────────
    if UNSLOTH:
        model, tokenizer = load_model_unsloth(args)
        peft_config = None          # LoRA already applied by unsloth
        use_liger = False           # unsloth handles its own kernel optimisations
        grad_ckpt = False           # handled by unsloth's get_peft_model
        grad_ckpt_kwargs = None
        # Qwen3.5 returns a VLProcessor (vision-language) even for text-only use.
        # Extract the underlying text tokenizer if needed.
        if not hasattr(tokenizer, 'convert_tokens_to_ids') and hasattr(tokenizer, 'tokenizer'):
            tokenizer = tokenizer.tokenizer

        # Unsloth's SFTTrainer wrapper sets args.eos_token='<EOS_TOKEN>' and
        # args.pad_token='<PAD_TOKEN>' — LLaMA-ism placeholders not in Qwen2 vocab.
        # TRL validates both → ValueError. Fix: register them in added_tokens_encoder
        # (a plain str→int dict, fully picklable, no closures, no method patching).
        # Training data uses <|im_end|> via chat template — these aliases are inert.
        _eos_id = int(tokenizer.convert_tokens_to_ids(tokenizer.eos_token))
        _pad_id = (int(tokenizer.convert_tokens_to_ids(tokenizer.pad_token))
                   if tokenizer.pad_token else _eos_id)
        tokenizer.added_tokens_encoder.setdefault("<EOS_TOKEN>", _eos_id)
        tokenizer.added_tokens_encoder.setdefault("<PAD_TOKEN>", _pad_id)
    else:
        model, tokenizer = load_model_hf(args)
        from peft import LoraConfig, TaskType
        peft_config = LoraConfig(
            r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, bias="none",
            task_type=TaskType.CAUSAL_LM, target_modules=LORA_TARGETS,
        )
        use_liger = True
        grad_ckpt = True
        grad_ckpt_kwargs = {"use_reentrant": False}

    # ── Dataset ────────────────────────────────────────────────────────────────
    dataset = load_dataset("json", data_files={
        "train": str(data_dir / "train.jsonl"),
        "validation": str(data_dir / "valid.jsonl"),
    })
    print(f"Dataset: {len(dataset['train'])} train / {len(dataset['validation'])} valid")

    # Unsloth requires chat template pre-applied — formatting_func conflicts with
    # completion_only_loss. Pre-apply here so both can work together.
    if UNSLOTH:
        def apply_template(batch):
            batch["text"] = [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in batch["messages"]
            ]
            return batch
        dataset = dataset.map(apply_template, batched=True, remove_columns=["messages"])
        print("Applied chat template to dataset (unsloth mode)")

    # ── SFTConfig ──────────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        weight_decay=args.weight_decay,
        optim="adamw_torch_fused",
        bf16=True,
        gradient_checkpointing=grad_ckpt,
        **({"gradient_checkpointing_kwargs": grad_ckpt_kwargs} if grad_ckpt_kwargs else {}),
        completion_only_loss=True,
        max_length=args.max_seq_len,
        **({"dataset_text_field": "text"} if UNSLOTH else {}),
        save_strategy="steps",
        save_steps=args.save_every,
        save_total_limit=None,
        eval_strategy="steps",
        eval_steps=args.eval_every,
        load_best_model_at_end=False,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=args.dataloader_workers,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=4,
        remove_unused_columns=False,
        dataset_num_proc=args.dataloader_workers,
        use_liger_kernel=use_liger,
    )

    # ── SFTTrainer ─────────────────────────────────────────────────────────────
    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = SFTTrainer(**trainer_kwargs)

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    # ── Train ──────────────────────────────────────────────────────────────────
    if args.resume_from:
        print(f"\nResuming from: {args.resume_from}")
        trainer.train(resume_from_checkpoint=args.resume_from)
    else:
        trainer.train()

    print(f"\nDone. Checkpoints: {args.output_dir}/checkpoint-{{200,400,...}}")
    print(f"Next: python3 finetune/sweep_v14.py")


if __name__ == "__main__":
    main()
