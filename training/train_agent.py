"""
training/train_agent.py — SQL Database Engineer Agent
Unsloth + GRPO training script.
Run on venue GPU (April 25-26) with compute credits.
ENV_URL points to live HF Space environment.
"""

import os
import json
import requests
from datasets import Dataset

# ── Try importing Unsloth (GPU only) ─────────────────────────
try:
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️  Unsloth not available. Run: pip install unsloth trl")

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

ENV_URL    = os.getenv("ENV_URL",    "https://junaid0600-sql-db-engineer-agent.hf.space")
HF_TOKEN   = os.getenv("HF_TOKEN",  "")
MODEL_NAME = os.getenv("MODEL_NAME", "unsloth/Qwen2.5-7B-Instruct")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./sdea-trained")

SYSTEM_PROMPT = """You are a senior database engineer. 
Given the current database state with slow queries, choose the BEST action to improve performance.
Think step by step:
1. If you haven't inspected queries yet → use inspect_query
2. If you haven't analyzed indexes → use analyze_indexes  
3. If you know which index is missing → use create_index
4. If query can be rewritten better → use rewrite_query
5. If table is huge (1M+ rows) → use partition_table
6. When performance target is reached → use submit_report

Respond with JSON only — no explanation, no markdown:
{"action_type": "...", "payload": {...}}"""


# ─────────────────────────────────────────────
#  REWARD FUNCTION (calls live HF Space)
# ─────────────────────────────────────────────

def reward_fn(prompts, completions, **kwargs):
    """
    GRPO reward function — calls /step on live environment.
    Returns list of float rewards, one per completion.
    """
    rewards = []
    task_ids = kwargs.get("task_ids", ["easy_s001"] * len(prompts))

    for i, (prompt, completion) in enumerate(zip(prompts, completions)):
        try:
            # Parse action from model output
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            text = text.strip().replace("```json", "").replace("```", "").strip()
            action = json.loads(text)

            # Reset environment for this task
            task_id = task_ids[i] if i < len(task_ids) else "easy_s001"
            requests.post(f"{ENV_URL}/reset",
                json={"task_id": task_id}, timeout=15)

            # Submit action and get reward
            resp = requests.post(f"{ENV_URL}/step",
                json=action, timeout=15)
            data = resp.json()
            score = data.get("reward", {}).get("score", 0.001)
            rewards.append(float(score))

        except json.JSONDecodeError:
            rewards.append(0.001)  # Invalid JSON output
        except Exception as e:
            print(f"Reward fn error: {e}")
            rewards.append(0.001)

    return rewards


# ─────────────────────────────────────────────
#  BUILD TRAINING DATASET
# ─────────────────────────────────────────────

def build_dataset():
    """Build training examples from all 15 Round 2 scenarios."""
    scenarios = []

    # Load all scenario files
    for fname in ["dataset/easy_scenarios.json",
                  "dataset/medium_scenarios.json",
                  "dataset/hard_scenarios.json"]:
        try:
            with open(fname) as f:
                scenarios.extend(json.load(f))
        except FileNotFoundError:
            print(f"{fname} not found, skipping")

    if not scenarios:
        # Fallback: fetch from live environment
        resp = requests.get(f"{ENV_URL}/tasks", timeout=15)
        tasks = resp.json().get("tasks", [])
        scenarios = [{"id": t["id"], "description": t["description"]} for t in tasks]

    examples = []
    for s in scenarios:
        prompt = f"""{SYSTEM_PROMPT}

Current Database State:
- Scenario: {s.get('id', 'unknown')}
- Description: {s.get('description', '')}
- Tables: {json.dumps(s.get('tables', []))}
- Slow Queries: {json.dumps(s.get('slow_queries', []))}
- Performance Score: {s.get('performance_score_baseline', 0)} / 100
- Target Score: {s.get('target_score', 85)}

What is your next action?"""

        examples.append({
            "prompt":   prompt,
            "task_id":  s.get("id", "easy_s001"),
        })

    print(f"Built {len(examples)} training examples")
    return Dataset.from_list(examples)


# ─────────────────────────────────────────────
#  MAIN TRAINING
# ─────────────────────────────────────────────

def train():
    if not UNSLOTH_AVAILABLE:
        print("Cannot train — Unsloth not installed")
        print("Run: pip install unsloth trl transformers datasets accelerate")
        return

    print(f"🚀 Loading model: {MODEL_NAME}")
    print(f"🌐 Environment: {ENV_URL}")

    # Load model with Unsloth 4-bit quantization
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = 4096,
        load_in_4bit   = True,
        token          = HF_TOKEN or None,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r              = 16,
        lora_alpha     = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_dropout   = 0,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",
    )

    # Build dataset
    dataset = build_dataset()

    # GRPO config
    config = GRPOConfig(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = 3,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 8,
        learning_rate               = 5e-5,
        max_completion_length       = 256,
        num_generations             = 4,
        logging_steps               = 10,
        save_steps                  = 50,
        warmup_ratio                = 0.1,
        report_to                   = "none",
    )

    # Reward function wrapper
    def reward_wrapper(prompts, completions, **kwargs):
        task_ids = [ex.get("task_id", "easy_s001") for ex in kwargs.get("batch", [])]
        return reward_fn(prompts, completions, task_ids=task_ids)

    # Train
    trainer = GRPOTrainer(
        model        = model,
        tokenizer    = tokenizer,
        reward_funcs = reward_wrapper,
        args         = config,
        train_dataset = dataset,
    )

    print("🏋️  Starting GRPO training...")
    trainer.train()

    # Save
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
    print(f"Training complete. Model saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    train()
