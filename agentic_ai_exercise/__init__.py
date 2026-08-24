from pathlib import Path

script_dir = Path(__file__).parent
ENV_PATH = script_dir / ".." / ".env"

QWEN3_VL_4B_Instruct = "Qwen/Qwen3-VL-4B-Instruct-FP8"
QWEN3_06B_Embed = "Qwen/Qwen3-Embedding-0.6B"