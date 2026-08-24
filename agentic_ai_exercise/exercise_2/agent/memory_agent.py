import os

from agentscope.agent import AgentBase
from agentscope.embedding import OpenAITextEmbedding
from agentscope.memory import Mem0LongTermMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from dotenv import load_dotenv
from mem0.vector_stores.configs import VectorStoreConfig
from agentic_ai_exercise import ENV_PATH, QWEN3_VL_4B_Instruct, QWEN3_06B_Embed

MODEL_CHAT_BOT = QWEN3_VL_4B_Instruct
MODEL_MEMORY = QWEN3_VL_4B_Instruct
EMBEDDING_MODEL = QWEN3_06B_Embed
EMBEDDING_MODEL_DIMENSIONS = 1024  

load_dotenv(ENV_PATH)
api_key = os.environ['LLM_API_KEY']
api_base = os.environ['LLM_BASE_URL']

def build_chat_model() -> OpenAIChatModel:
    """Instantiate and return the chat model used for answer generation.

    Returns:
        Configured OpenAIChatModel instance.
    """
    return OpenAIChatModel(
        model_name=MODEL_CHAT_BOT,
        api_key=api_key,
        client_kwargs={'base_url': api_base},
        stream=False,
        generate_kwargs={"max_tokens": 512},
    )


def create_memory_instance() -> Mem0LongTermMemory:
    """Instantiate and return a Mem0LongTermMemory backed by an in-memory Qdrant store.

    Returns:
        Configured Mem0LongTermMemory instance.
    """
    long_term_memory = Mem0LongTermMemory(
            agent_name="Memor",
            user_name="user_123",
            model=OpenAIChatModel(
                model_name=MODEL_MEMORY,
                api_key=api_key,
                client_kwargs={'base_url': api_base},
                stream=False
            ),
            embedding_model=OpenAITextEmbedding(
                model_name=QWEN3_06B_Embed,
                api_key=api_key,
                base_url=api_base,
                dimensions=None,
            ),
            vector_store_config=VectorStoreConfig(
                provider='qdrant',
                config={
                    "embedding_model_dims": EMBEDDING_MODEL_DIMENSIONS,
                    "on_disk": False,
                }
            )
        )
    return long_term_memory
    
    
class MemoryAgent(AgentBase):
    """An agent that uses Mem0LongTermMemory to persist and retrieve knowledge.

    On each reply the agent (1) retrieves relevant memories from the vector
    store, (2) generates an answer using the LLM, and (3) records the Q&A
    exchange back into memory for future retrieval.

    Attributes:
        name: Display name of the agent.
        model: Chat model used for answer generation.
        memory: Long-term memory instance backed by Qdrant.
    """

    def __init__(self, name: str) -> None:
        """Initialise the agent with a fresh chat model and memory instance.

        Args:
            name: Display name of the agent.
        """
        super().__init__()
        self.name = name
        self.model = build_chat_model()
        self.memory = create_memory_instance()

    async def reply(self, msg: Msg, system_prompt: str | None = None) -> Msg:
        """Generate a reply to *msg*, augmented with retrieved memories.

        Args:
            msg: Incoming user message.
            system_prompt: Optional system-level instruction prepended to the
                prompt (e.g. answer-format constraints). Applied before any
                retrieved memories.

        Returns:
            Assistant message containing the model's response.
        """
        # Step 1 -> retrieve memories (errors are non-fatal)
        try:
            memories = await self.memory.retrieve([msg])
        except Exception:
            memories = None
        # Step 2 -> given the retrieved memories build the new prompt
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        if memories:
            messages.append({'role': 'system', 'content': f"Relevant memories: {memories}"})
        messages.append({'role': 'user', 'content': msg.content})

        response = await self.model(messages)

        wrapped_response = Msg(
            name=self.name,
            role="assistant",
            content=response.content,
        )

        # Step 3 -> save into the memory the exchange (errors are non-fatal)
        try:
            await self.memory.record([msg, wrapped_response])
        except Exception:
            pass

        return wrapped_response


        