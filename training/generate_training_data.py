"""
training/generate_training_data.py
Generates training data by running episodes on the live environment.
Saves (prompt, action, reward) tuples for GRPO training.
Run this BEFORE train_agent.py to pre-generate data.
"""

import os
import json
import time
import requests
from pathlib import Path

ENV_URL    = os.getenv("ENV_URL", "https://junaid0600-sql-db-engineer-agent.hf.space")
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./sdea-trained"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# All Round 2 scenario IDs
ALL_SCENARIOS = [
    "easy_s001", "easy_s002", "easy_s003", "easy_s004", "easy_s005",
    "medium_s001", "medium_s002", "medium_s003", "medium_s004", "medium_s005",
    "hard_s001", "hard_s002", "hard_s003", "hard_s004", "hard_s005",
]

# Optimal action sequences per scenario (expert demonstrations)
EXPERT_ACTIONS = {
    "easy_s001": [
        {"action_type": "inspect_query",    "payload": {"query_id": "q1"}},
        {"action_type": "analyze_indexes",  "payload": {"table": "users"}},
        {"action_type": "create_index",     "payload": {"table": "users", "columns": ["email"]}},
        {"action_type": "submit_report",    "payload": {"summary": "Added index on users(email). Email lookup now uses index scan instead of full table scan."}},
    ],
    "easy_s002": [
        {"action_type": "inspect_query",   "payload": {"query_id": "q1"}},
        {"action_type": "create_index",    "payload": {"table": "orders", "columns": ["user_id", "status"]}},
        {"action_type": "submit_report",   "payload": {"summary": "Composite index on orders(user_id, status) eliminates full table scan."}},
    ],
    "easy_s003": [
        {"action_type": "inspect_query",   "payload": {"query_id": "q1"}},
        {"action_type": "create_index",    "payload": {"table": "products", "columns": ["name"]}},
        {"action_type": "submit_report",   "payload": {"summary": "Index on products(name) speeds up LIKE queries."}},
    ],
    "easy_s004": [
        {"action_type": "inspect_query",   "payload": {"query_id": "q1"}},
        {"action_type": "create_index",    "payload": {"table": "sessions", "columns": ["user_id", "expires_at"]}},
        {"action_type": "submit_report",   "payload": {"summary": "Composite index on sessions(user_id, expires_at) added."}},
    ],
    "easy_s005": [
        {"action_type": "inspect_query",   "payload": {"query_id": "q1"}},
        {"action_type": "create_index",    "payload": {"table": "logs", "columns": ["level", "created_at"]}},
        {"action_type": "submit_report",   "payload": {"summary": "Compound index on logs(level, created_at) added."}},
    ],
    "medium_s001": [
        {"action_type": "inspect_query",      "payload": {"query_id": "q1"}},
        {"action_type": "inspect_query",      "payload": {"query_id": "q2"}},
        {"action_type": "analyze_indexes",    "payload": {"table": "orders"}},
        {"action_type": "create_index",       "payload": {"table": "orders", "columns": ["user_id", "status"]}},
        {"action_type": "create_index",       "payload": {"table": "users",  "columns": ["country"]}},
        {"action_type": "analyze_statistics", "payload": {"table": "orders"}},
        {"action_type": "submit_report",      "payload": {"summary": "Two indexes added. Both slow queries now use index scans."}},
    ],
    "medium_s002": [
        {"action_type": "inspect_query",   "payload": {"query_id": "q1"}},
        {"action_type": "inspect_query",   "payload": {"query_id": "q2"}},
        {"action_type": "create_index",    "payload": {"table": "posts",   "columns": ["author_id", "published", "created_at"]}},
        {"action_type": "create_index",    "payload": {"table": "authors", "columns": ["username"]}},
        {"action_type": "submit_report",   "payload": {"summary": "Multi-column index on posts and unique index on authors added."}},
    ],
    "medium_s003": [
        {"action_type": "inspect_query",      "payload": {"query_id": "q1"}},
        {"action_type": "inspect_query",      "payload": {"query_id": "q2"}},
        {"action_type": "create_index",       "payload": {"table": "stock_movements", "columns": ["product_id", "movement_type", "created_at"]}},
        {"action_type": "rewrite_query",      "payload": {"query_id": "q2", "new_sql": "SELECT p.id, p.name, SUM(sm.quantity) FROM products p INNER JOIN stock_movements sm ON p.id = sm.product_id GROUP BY p.id, p.name"}},
        {"action_type": "analyze_statistics", "payload": {"table": "stock_movements"}},
        {"action_type": "submit_report",      "payload": {"summary": "Index + query rewrite applied. Implicit join converted to explicit INNER JOIN."}},
    ],
    "medium_s004": [
        {"action_type": "inspect_query",   "payload": {"query_id": "q1"}},
        {"action_type": "inspect_query",   "payload": {"query_id": "q2"}},
        {"action_type": "analyze_indexes", "payload": {"table": "tickets"}},
        {"action_type": "create_index",    "payload": {"table": "tickets", "columns": ["status", "priority", "created_at"]}},
        {"action_type": "create_index",    "payload": {"table": "tickets", "columns": ["status", "agent_id"]}},
        {"action_type": "submit_report",   "payload": {"summary": "Two targeted indexes on tickets table eliminate full table scans."}},
    ],
    "medium_s005": [
        {"action_type": "inspect_query",      "payload": {"query_id": "q1"}},
        {"action_type": "inspect_query",      "payload": {"query_id": "q2"}},
        {"action_type": "create_index",       "payload": {"table": "events", "columns": ["user_id", "event_type", "occurred_at"]}},
        {"action_type": "create_index",       "payload": {"table": "users",  "columns": ["signup_source", "created_at"]}},
        {"action_type": "analyze_statistics", "payload": {"table": "events"}},
        {"action_type": "submit_report",      "payload": {"summary": "Range query indexes added for both events and users tables."}},
    ],
    "hard_s001": [
        {"action_type": "inspect_query",      "payload": {"query_id": "q1"}},
        {"action_type": "inspect_query",      "payload": {"query_id": "q2"}},
        {"action_type": "inspect_query",      "payload": {"query_id": "q3"}},
        {"action_type": "analyze_indexes",    "payload": {"table": "transactions"}},
        {"action_type": "create_index",       "payload": {"table": "transactions", "columns": ["account_id", "status", "created_at"]}},
        {"action_type": "create_index",       "payload": {"table": "transactions", "columns": ["customer_id", "amount"]}},
        {"action_type": "create_index",       "payload": {"table": "audit_log",    "columns": ["entity_id", "entity_type", "created_at"]}},
        {"action_type": "rewrite_query",      "payload": {"query_id": "q2", "new_sql": "SELECT c.id, c.name, COUNT(t.id) as tx_count FROM customers c INNER JOIN transactions t ON c.id = t.customer_id WHERE t.amount > ? GROUP BY c.id, c.name"}},
        {"action_type": "partition_table",    "payload": {"table": "audit_log", "partition_by": "created_at", "partition_type": "RANGE"}},
        {"action_type": "analyze_statistics", "payload": {"table": "transactions"}},
        {"action_type": "submit_report",      "payload": {"summary": "3 indexes added, implicit join rewritten, audit_log partitioned by date."}},
    ],
}


def run_expert_episode(scenario_id: str) -> list[dict]:
    """
    Run one expert episode and collect (prompt, action, reward) tuples.
    """
    trajectory = []

    try:
        # Reset
        r = requests.post(f"{ENV_URL}/reset",
            json={"task_id": scenario_id}, timeout=15)
        obs = r.json()

        actions = EXPERT_ACTIONS.get(scenario_id, [
            {"action_type": "inspect_query",   "payload": {"query_id": "q1"}},
            {"action_type": "create_index",    "payload": {"table": "orders", "columns": ["user_id"]}},
            {"action_type": "submit_report",   "payload": {"summary": "Optimization applied."}},
        ])

        for action in actions:
            # Build prompt from current observation
            ctx   = obs.get("current_context", {})
            prompt = f"""You are a senior database engineer.
Current DB state:
- Performance score: {ctx.get('performance_score', 0)} / {ctx.get('target_score', 85)}
- Slow queries: {json.dumps(ctx.get('slow_queries', []))}
- Tables: {json.dumps(ctx.get('tables', []))}
- Steps remaining: {obs.get('max_steps', 50) - obs.get('step_count', 0)}
Choose the best next action as JSON:"""

            # Take action
            r    = requests.post(f"{ENV_URL}/step", json=action, timeout=15)
            data = r.json()

            trajectory.append({
                "scenario_id": scenario_id,
                "prompt":      prompt,
                "action":      json.dumps(action),
                "reward":      data.get("reward", {}).get("score", 0.001),
                "db_delta":    data.get("info", {}).get("db_delta", 0),
                "step":        data.get("observation", {}).get("step_count", 0),
            })

            obs = data.get("observation", obs)
            if data.get("done"):
                break

        print(f"  ✅ {scenario_id}: {len(trajectory)} steps, "
              f"final reward={trajectory[-1]['reward']:.3f}")

    except Exception as e:
        print(f"  ❌ {scenario_id}: {e}")

    return trajectory


def generate_all():
    """Generate training data from all scenarios."""
    print("🔄 Generating training data...")
    print(f"🌐 Environment: {ENV_URL}")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("─" * 50)

    all_trajectories = []
    total_steps      = 0

    for scenario_id in ALL_SCENARIOS:
        print(f"Running {scenario_id}...")
        trajectory = run_expert_episode(scenario_id)
        all_trajectories.extend(trajectory)
        total_steps += len(trajectory)
        time.sleep(0.5)  # Be nice to HF Space

    # Save as JSONL for training
    output_file = OUTPUT_DIR / "training_data.jsonl"
    with open(output_file, "w") as f:
        for item in all_trajectories:
            f.write(json.dumps(item) + "\n")

    # Also save as JSON for inspection
    json_file = OUTPUT_DIR / "training_data.json"
    with open(json_file, "w") as f:
        json.dump(all_trajectories, f, indent=2)

    print("─" * 50)
    print(f"✅ Generated {total_steps} training steps from {len(ALL_SCENARIOS)} scenarios")
    print(f"📄 Saved to: {output_file}")

    # Stats
    rewards = [t["reward"] for t in all_trajectories]
    avg_r   = sum(rewards) / max(len(rewards), 1)
    print(f"📊 Average reward: {avg_r:.3f}")
    print(f"📊 Max reward:     {max(rewards):.3f}")
    print(f"📊 Min reward:     {min(rewards):.3f}")

    return all_trajectories


if __name__ == "__main__":
    generate_all()
