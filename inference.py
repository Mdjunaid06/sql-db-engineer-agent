# inference.py — required by OpenEnv validator
from baseline import run_baseline

# Entry point for openenv validator
app = None  # placeholder for server entry point

def main():
    print("Running inference / baseline agent...")
    response = run_baseline()
    print(f"\nInference Results:")
    for r in response.results:
        print(f"  {r.difficulty.value:8} | {r.task_id:12} | score={r.score} | steps={r.steps}")
    print(f"\nAverage Score: {response.average_score}")

if __name__ == "__main__":
    main()