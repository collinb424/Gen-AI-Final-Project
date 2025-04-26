import chainlit as cl
from langchain.agents import create_react_agent, tool
# from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import PromptTemplate
from llm import LLM
from vectorstore import VECTOR_STORE
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents import AgentExecutor

# Moving onto Agentic RAG
@tool(response_format="content")
async def retrieve(q: str):
    """Retrieve information related to a query."""

    retrieved_docs = VECTOR_STORE.similarity_search(q, k = 5)
    doc_strings = [
        f"## Source: {doc.metadata}\n### Content: {doc.page_content}"
        for doc in retrieved_docs
    ]
    context = "\n\n".join(doc_strings)
    return context


# ReAct prompt setup
REACT_PROMPT_TEMPLATE = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}'''

REACT_PROMPT = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

# MEMORY = InMemorySaver()
AGENT = create_react_agent(LLM, [retrieve], REACT_PROMPT)
agent_executor = AgentExecutor(agent=AGENT, tools=[retrieve], verbose=True, handle_parsing_errors=True)


# Chainlink
@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}

    user_input = {
        "input": message.content,
    }

    async for step in agent_executor.astream(user_input, config=config):
        if hasattr(step, "log"):
            await cl.Message(content=step.log).send()
        elif hasattr(step, "return_values"):
            await cl.Message(content=step.return_values["output"]).send()
        else:
            await cl.Message(content=str(step)).send()