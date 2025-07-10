# Langchain Imports
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from langchain_core.messages import BaseMessage
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory,
)
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Other system imports
from dotenv import load_dotenv

# Pydantic Models
from pydantic import BaseModel, Field


load_dotenv()


llm = ChatOpenAI(model="o3-mini-2025-01-31")


class BufferWindowMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    k: int = Field(default_factory=int)

    def __init__(self, k: int):
        super().__init__(k=k)
        print(f"Initializing BufferWindowMessageHistory with k={k}")

    def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages to the history, removing any messages beyond
        the last `k` messages.
        """
        self.messages.extend(messages)
        self.messages = self.messages[-self.k :]

    def clear(self) -> None:
        """Clear the history."""
        self.messages = []


chat_map = {}


def get_chat_history(session_id: str, k: int = 4) -> BufferWindowMessageHistory:
    if session_id not in chat_map:
        # if session ID doesn't exist, create a new chat history
        chat_map[session_id] = BufferWindowMessageHistory(k)
    return chat_map[session_id]


system_prompt = "You are a helpful assistant called Zeta."

prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

pipeline = prompt_template | llm
pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="The session ID to use for the chat history",
            default="id_default",
        ),
        ConfigurableFieldSpec(
            id="k",
            annotation=int,
            name="k",
            description="The number of messages to keep in the history",
            default=4,
        ),
    ],
)

# RAG-> retreive -> []
response = pipeline_with_history.invoke(
    {"query": "Hi, my name is Josh"},
    config={"configurable": {"session_id": "1", "k": 1}},
)

print(f"Chat Model Response: {response.content} ")
print(f'Message History: {chat_map["1"]}')

# HW
# asyncio
# Python generators
# 1. Runnable indepth -> | pipe, batch, async, stream vs invoke
# 2. Revise
# Bonus Task: Implement summarization in message history
