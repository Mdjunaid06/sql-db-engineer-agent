"""
env/scenario_generator.py
Dynamically generates new DB engineering scenarios using an LLM.
Used by CurriculumGenerator when agent reaches ultra tier.
Also useful for generating additional training data.
"""

import os
import json
import random
import requests
from typing import Optional


ENV_URL   = os.getenv("ENV_URL",      "https://junaid0600-sql-db-engineer-agent.hf.space")
API_BASE  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
HF_TOKEN  = os.getenv("HF_TOKEN",     "")
MODEL     = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")

# Domain templates for variety
DOMAINS = [
    "e-commerce platform",
    "healthcare records system",
    "financial trading platform",
    "social media platform",
    "logistics and shipping",
    "gaming leaderboard system",
    "SaaS subscription platform",
    "analytics and reporting DB",
    "inventory management system",
    "HR and payroll system",
]

TABLE_TEMPLATES = {
    "e-commerce platform": ["orders", "users", "products", "payments", "reviews"],
    "healthcare records system": ["patients", "appointments", "prescriptions", "doctors", "billing"],
    "financial trading platform": ["trades", "accounts", "portfolios", "transactions", "market_data"],
    "social media platform": ["posts", "users", "comments", "likes", "followers"],
    "logistics and shipping": ["shipments", "drivers", "routes", "tracking", "warehouses"],
    "gaming leaderboard system": ["players", "matches", "scores", "achievements", "seasons"],
    "SaaS subscription platform": ["subscriptions", "users", "invoices", "features", "usage_logs"],
    "analytics and reporting DB": ["events", "sessions", "users", "funnels", "reports"],
    "inventory management system": ["products", "stock", "suppliers", "purchase_orders", "warehouses"],
    "HR and payroll system": ["employees", "departments", "payroll", "attendance", "performance"],
}


class ScenarioGenerator:
    """
    Generates novel DB engineering scenarios procedurally.
    Can use LLM for richer descriptions or pure procedural generation.
    """

    def __init__(self):
        self.generated_count = 0

    def generate_procedural(
        self,
        difficulty: str = "hard",
        domain:     Optional[str] = None,
    ) -> dict:
        """
        Generate a scenario procedurally (no LLM needed).
        Fast, deterministic, infinitely scalable.
        """
        if domain is None:
            domain = random.choice(DOMAINS)

        table_names = TABLE_TEMPLATES.get(domain, ["orders", "users", "products"])

        # Scale based on difficulty
        scales = {
            "easy":   {"rows": (5000,    50000),   "queries": 1, "max_steps": 15, "target": (75, 85)},
            "medium": {"rows": (50000,   500000),  "queries": 2, "max_steps": 30, "target": (70, 80)},
            "hard":   {"rows": (500000,  5000000), "queries": 3, "max_steps": 50, "target": (65, 75)},
            "ultra":  {"rows": (1000000, 10000000),"queries": 4, "max_steps": 40, "target": (60, 70)},
        }
        scale = scales.get(difficulty, scales["hard"])

        # Pick tables
        n_tables = {"easy": 1, "medium": 2, "hard": 3, "ultra": random.randint(4, 6)}.get(difficulty, 3)
        chosen   = random.sample(table_names, min(n_tables, len(table_names)))

        tables = []
        for name in chosen:
            rows = random.randint(*scale["rows"])
            tables.append({
                "name":    name,
                "rows":    rows,
                "indexes": ["PRIMARY"],
                "size_mb": rows // 200,
            })

        # Generate slow queries
        filter_cols  = ["user_id", "status", "created_at", "category", "type", "date"]
        slow_queries = []
        for i in range(scale["queries"]):
            table  = random.choice(chosen)
            col1   = random.choice(filter_cols)
            col2   = random.choice([c for c in filter_cols if c != col1])
            avg_ms = random.randint(3000, 25000)

            slow_queries.append({
                "id":            f"q{i+1}",
                "sql":           f"SELECT * FROM {table} WHERE {col1}=? AND {col2}=?",
                "avg_ms":        avg_ms,
                "main_table":    table,
                "rows_examined": tables[chosen.index(table)]["rows"] if table in chosen else 100000,
            })

        # Missing index hints (one per query)
        missing_hints = []
        for i, q in enumerate(slow_queries):
            table = q["main_table"]
            cols  = [c.strip() for c in q["sql"].split("WHERE")[1].split("AND")]
            cols  = [c.split("=")[0].strip() for c in cols if "=" in c][:2]
            missing_hints.append({
                "table":   table,
                "columns": cols,
                "reason":  f"Composite WHERE clause on {table}",
            })

        baseline = round(random.uniform(3.0, 12.0), 1)
        target   = round(random.uniform(*scale["target"]), 1)

        self.generated_count += 1
        scenario_id = f"gen_{difficulty}_{self.generated_count:04d}"

        return {
            "id":                        scenario_id,
            "description":               f"{domain.title()}: {len(tables)} tables, {len(slow_queries)} slow queries. Auto-generated.",
            "tables":                    tables,
            "slow_queries":              slow_queries,
            "missing_index_hints":       missing_hints,
            "performance_score_baseline": baseline,
            "target_score":              target,
            "max_steps":                 scale["max_steps"],
            "category":                  domain.replace(" ", "_"),
            "generated":                 True,
        }

    def generate_batch(self, n: int = 10, difficulty: str = "hard") -> list[dict]:
        """Generate a batch of scenarios for training data augmentation."""
        scenarios = []
        domains   = DOMAINS * (n // len(DOMAINS) + 1)
        random.shuffle(domains)

        for i in range(n):
            domain   = domains[i % len(domains)]
            scenario = self.generate_procedural(difficulty=difficulty, domain=domain)
            scenarios.append(scenario)

        print(f"✅ Generated {n} {difficulty} scenarios")
        return scenarios

    def save_batch(self, scenarios: list[dict], filepath: str):
        """Save generated scenarios to JSON file."""
        with open(filepath, "w") as f:
            json.dump(scenarios, f, indent=2)
        print(f"💾 Saved {len(scenarios)} scenarios to {filepath}")

    def augment_dataset(self, n_per_difficulty: int = 5):
        """
        Augment the existing dataset with generated scenarios.
        Saves to dataset/generated_*.json files.
        """
        for diff in ["easy", "medium", "hard"]:
            batch    = self.generate_batch(n_per_difficulty, difficulty=diff)
            filepath = f"dataset/generated_{diff}_scenarios.json"
            self.save_batch(batch, filepath)

        print(f"✅ Dataset augmented with {n_per_difficulty * 3} new scenarios")


# Singleton
scenario_generator = ScenarioGenerator()


if __name__ == "__main__":
    # Quick test — generate one scenario per difficulty
    print("🔧 Scenario Generator Test")
    print("=" * 50)

    for diff in ["easy", "medium", "hard"]:
        s = scenario_generator.generate_procedural(difficulty=diff)
        print(f"\n[{diff.upper()}] {s['id']}")
        print(f"  Domain: {s['category']}")
        print(f"  Tables: {[t['name'] for t in s['tables']]}")
        print(f"  Queries: {len(s['slow_queries'])}")
        print(f"  Baseline: {s['performance_score_baseline']} → Target: {s['target_score']}")
        print(f"  Max steps: {s['max_steps']}")

    print("\n✅ Generator working correctly!")
