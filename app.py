import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from laya_ignite.app import launch_ui

if __name__ == "__main__":
    launch_ui(port=7860, share=False)
