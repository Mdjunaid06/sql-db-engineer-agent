# server/app.py — required by OpenEnv validator
import uvicorn
from api.server import app

def main():
    """Main entry point required by OpenEnv validator."""
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()