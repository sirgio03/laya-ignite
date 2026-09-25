import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure ZeroGPU detection on Hugging Face Spaces entrypoint
try:
    import spaces
    @spaces.GPU
    def _zero_gpu_check():
        """Satisfies Hugging Face ZeroGPU startup scanner."""
        return True
except Exception:
    pass

from laya_ignite.app import demo

if __name__ == "__main__":
    demo.launch(server_port=7860, share=False)
