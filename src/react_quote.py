from typing import Annotated, TypedDict, List
from llm import LLM
from langchain.prompts import PromptTemplate
from vectorstore import VECTOR_STORE
from langchain.schema.runnable import Runnable
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from util import extract_year
import chainlit as cl
from langchain.agents import create_react_agent, AgentExecutor, tool
from langchain_core.agents import AgentAction, AgentFinish

# Keep the Citation and QuotedAnswer models the same
class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )

class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )
    sources: Annotated[
        List[str],
        ...,
        "List of sources (author + page number) used to answer the question",
    ]

# Keep the format_docs_with_id function the same
def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\n"
        f"Article Title: {doc.metadata.get('title', 'Untitled')}\n"
        f"Author: {doc.metadata.get('author', 'Unknown')}\n"
        f"Page Number: {int(doc.metadata.get('page', 0)) + 1}\n"
        f"Year: {extract_year(doc.metadata.get('creationdate', ''))}\n"
        f"Article Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ] 
    for i, doc in enumerate(docs): print(doc.metadata)
    return "\n\n" + "\n\n".join(formatted)

# Define a global variable to store retrieved documents
retrieved_documents = []

# Create a tool for retrieval instead of a function
@tool(response_format="content")
def retrieve_documents(query: str):
    """Retrieve information related to a query from the document store."""
    global retrieved_documents
    retrieved_documents = VECTOR_STORE.similarity_search(query, k=5)
    formatted_docs = format_docs_with_id(retrieved_documents)
    return f"Retrieved {len(retrieved_documents)} relevant documents:\n{formatted_docs}"

# Create a tool for generating the final answer with citations
@tool(response_format="content")
def generate_quoted_answer(question: str):
    """Generate a well-cited answer based on the retrieved documents."""
    global retrieved_documents
    if not retrieved_documents:
        return "No documents have been retrieved yet. Please use the retrieve_documents tool first."
    
    formatted_docs = format_docs_with_id(retrieved_documents)
    prompt_text = """
    You're an expert in answering questions. Use the following pieces of documents relevant to the question to answer it. Make sure to not make up new information and only use the information provided in the documents. If you cannot find the relevant information from the documents, answer "I cannot find that information in the provided documents."

    For each point you make, include a relevant **verbatim quote** from the source documents *in-line*, using parentheses or blockquotes. For each claim, cite your source using the format (Author, Year, p. Page Number)."

    Question: {question}
    Context: {context}
    Answer:
    """.strip()
    
    prompt = PromptTemplate.from_template(prompt_text)
    messages = prompt.invoke({"question": question, "context": formatted_docs})
    structured_llm = LLM.with_structured_output(QuotedAnswer)
    response = structured_llm.invoke(messages)
    
    # Format the response with citations
    if not response.citations:
        return f"**Answer:** {response.answer}"
    else:
        citations_str = ""
        for idx, c in enumerate(response.citations, start=1):
            source_doc = retrieved_documents[c.source_id]
            author = source_doc.metadata.get("author", "Unknown")
            year = extract_year(source_doc.metadata.get('creationdate', ''))
            page = int(source_doc.metadata.get("page", 0)) + 1
            citations_str += (
                f'Quote {idx}: "{c.quote}" ({author}, {year}, p. {page})\n\n'
            )
        return f"**Answer:** {response.answer}\n\n**Citations:**\n{citations_str}"

# Create the ReAct prompt template
REACT_PROMPT_TEMPLATE = '''Answer the following question as best you can, making sure to provide well-cited answers with quotes from the documents. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question with proper citations

Begin!

Question: {input}
{agent_scratchpad}'''

REACT_PROMPT = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)

# Create the ReAct agent
tools = [retrieve_documents, generate_quoted_answer]
agent = create_react_agent(LLM, tools, REACT_PROMPT)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Chainlit integration
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Hi! Ask a question and I'll answer it with sources based on the provided documents.").send()

@cl.on_message
async def on_message(message: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    user_input = {"input": message.content}
    
    async for step in agent_executor.astream(user_input, config=config):
        if isinstance(step, dict) and "intermediate_steps" in step:
            # This is an intermediate step in the ReAct process
            # You could display the agent's thinking if desired
            pass
        elif hasattr(step, "log"):
            await cl.Message(content=step.log).send()
        elif hasattr(step, "return_values"):
            await cl.Message(content=step.return_values["output"]).send()
        else:
            # Only send final outputs to the user, not intermediate thinking
            if "Final Answer:" in str(step):
                await cl.Message(content=str(step).split("Final Answer:")[-1].strip()).send()