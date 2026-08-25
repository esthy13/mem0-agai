from project_config import QWEN3_06B_Embed, QWEN3_VL_4B_Instruct

MODEL_CHAT_BOT = QWEN3_VL_4B_Instruct
MODEL_MEMORY = QWEN3_VL_4B_Instruct
EMBEDDING_MODEL = QWEN3_06B_Embed
EMBEDDING_MODEL_DIMENSIONS = 1024  # native output dim of Qwen3-Embedding-0.6B


DATASET_TO_EVALUATE = "hotpot_qa"
