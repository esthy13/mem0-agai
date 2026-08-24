import asyncio
import os

from dotenv import load_dotenv

from agentscope.agent import AgentBase
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel

from agentic_ai_exercise import ENV_PATH, QWEN3_VL_4B_Instruct

# Load API key and base URL from .env
load_dotenv(ENV_PATH)

api_key  = os.environ["LLM_API_KEY"]
api_base = os.environ["LLM_BASE_URL"]


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
            stream=False
        )
        self.formatter = OpenAIChatFormatter()
        self.memory = InMemoryMemory()

    async def reply(self, msg: Msg | list[Msg] | None) -> Msg:
        """Core agent logic: store input, call model, store and return response."""
        await self.print(msg)
        # 1. Store the incoming user message
        await self.memory.add(msg)

        # 2. Build the prompt: system prompt + full conversation history
        prompt = await self.formatter.format(
            [
                Msg("system", self.sys_prompt, "system"),
                *await self.memory.get_memory(),
            ]
        )

        # 3. Call the model
        response = await self.model(prompt)

        # 4. Wrap the response in a Msg and store it
        reply_msg = Msg(
            name=self.name,
            content=response.content,
            role="assistant",
        )
        await self.memory.add(reply_msg)
        await self.print(reply_msg)

        return reply_msg


async def main() -> None:
    agent = MyAgent()

    # Multi-turn conversation
    turns = [
        "Hello! What can you help me with?",
        "Can you summarise that in one sentence?",
    ]
    for text in turns:
        user_msg = Msg(name="user", content=text, role="user")
        await agent.reply(user_msg)


if __name__ == "__main__":
    asyncio.run(main())