import asyncio
import os

from dotenv import load_dotenv

from agentscope.agent import AgentBase
from agentscope.embedding import OpenAITextEmbedding
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

from agentic_ai_exercise import ENV_PATH, QWEN3_06B_Embed, QWEN3_VL_4B_Instruct

# Load API key and base URL from .env
load_dotenv(ENV_PATH)

api_key  = os.environ["LLM_API_KEY"]
api_base = os.environ["LLM_BASE_URL"]


# --- Embedding model (standalone, used outside the agent) ---
embedding_model = OpenAITextEmbedding(
    model_name=QWEN3_06B_Embed,
    api_key=api_key,
    base_url=api_base,           # note: base_url, not client_args
    dimensions=None,
)


class MyAgent(AgentBase):
    """A simple base agent using OpenAIChatModel."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "Assistant"
        self.sys_prompt = "You are a helpful assistant."

        self.model = OpenAIChatModel(
            model_name=QWEN3_VL_4B_Instruct,
            api_key=api_key,
            client_args={"base_url": api_base},
            stream=False,
        )
        self.formatter = OpenAIChatFormatter()
        self.memory = InMemoryMemory()

    async def reply(self, msg: Msg | list[Msg] | None) -> Msg:
        await self.memory.add(msg)

        prompt = await self.formatter.format(
            [
                Msg("system", self.sys_prompt, "system"),
                *await self.memory.get_memory(),
            ]
        )

        response = await self.model(prompt)

        reply_msg = Msg(
            name=self.name,
            content=response.content,
            role="assistant",
        )
        await self.memory.add(reply_msg)
        await self.print(reply_msg)
        return reply_msg


async def main() -> None:
    # --- Embedding example ---
    texts = [
        "AgentScope makes building agents easy.",
        "Paris is the capital of France.",
    ]
    embed_response = await embedding_model(texts)
    print(f"Embedded {len(embed_response.embeddings)} texts")
    print(f"Embedding dim: {len(embed_response.embeddings[0])}")
    print(f"Usage: {embed_response.usage}\n")

    # --- Chat agent example ---
    agent = MyAgent()
    msg = Msg(name="user", content="Hello! What can you help me with?", role="user")
    await agent.reply(msg)


if __name__ == "__main__":
    asyncio.run(main())