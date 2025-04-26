from typing import Annotated, TypedDict
from llm import LLM
from langchain.prompts import PromptTemplate
from vectorstore import VECTOR_STORE
from langgraph.graph import StateGraph, START
from langchain.schema.runnable import Runnable
from pydantic import BaseModel, Field
from typing import List
from langchain_core.documents import Document
from util import extract_year, verify_quote_in_source
import chainlit as cl


# RAG
PROMPT = """
Context:
You are a subject matter expert tasked with answering a question based on the provided sources. The sources are excerpts from various documents relevant to the topic.

Role:
Write as an insightful academic guide. Maintain a clear, thoughtful, and precise tone. Your writing style should be analytical, balanced, and capable of comparing different perspectives if they exist.

Action:
- Only use information present in the provided documents. Do not invent or assume.
- If multiple sources agree, synthesize their insights.
- If sources differ, compare and contrast their approaches, noting similarities and differences explicitly.
- Incorporate direct **verbatim quotes** from the documents into your answer to substantiate key points. Again, to make it very clear, you should not just describe what the sources say or reference the quotes at the bottom, but use VERBATIM quotes throughout your answer.
- After each quote, cite the source in the format (Author, Year, p. Page Number). If you cannot get the Author, instead do (File Name, Page Number). If it is multiple authors, do (Author et al., Year, p. Page Number)
- If no information is available to answer the question, respond with "I cannot find that information in the provided documents."

Format:
Present your answer as a structured academic paragraph. 
- Embed quotations naturally into the explanation.
- Use comparison language when appropriate
- Ensure that each claim is supported by a specific quote.

Target:
Your audience is an educated reader who is looking for a clear, nuanced, and well-supported answer to their question based on real sources.

Question: {question}
Context: {context}
Answer:
""".strip()

PROMPT = PromptTemplate.from_template(PROMPT)

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


class State(TypedDict):
    question: str
    context: List[Document]
    answer: QuotedAnswer
    formatted: str



def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"File Name: {doc.metadata.get('source').split('/')[1]}\n"
        f"Article Title: {doc.metadata.get('title', 'Untitled')}\n"
        f"Author: {doc.metadata.get('author', 'Unknown')}\n"
        f"Page Number: {int(doc.metadata.get('page', 0)) + 1}\n"
        f"Year: {extract_year(doc.metadata.get('creationdate', ''))}\n"
        f"Article Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ] 
    for i, doc in enumerate(docs): print(doc.metadata)
    return "\n\n" + "\n\n".join(formatted)


def retrieve(state):
    results = VECTOR_STORE.similarity_search(state["question"], k=5)
    return {"context": results}

def generate(state):
    formatted_docs = format_docs_with_id(state["context"])
    messages = PROMPT.invoke({"question": state["question"], "context": formatted_docs})
    structured_llm = LLM.with_structured_output(QuotedAnswer)
    response = structured_llm.invoke(messages)
    return {"answer": response, "formatted": formatted_docs}



# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


# Chainlit integration
@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="Hi! Ask a question and I’ll answer it with sources based on the provided documents.").send()

@cl.on_message
async def on_message(message: cl.Message):
    question = message.content
    state = {"question": question}
    try:
        response = graph.invoke(state)
        quoted_answer = response["answer"]
        formatted_docs = response["formatted"]

        if not quoted_answer.citations:
            full_response = f"**Answer:** {quoted_answer.answer}"
        else:
            # Format the output with citations
            citations_str = ""
            for idx, c in enumerate(quoted_answer.citations, start=1):
                source_doc = response["context"][c.source_id]

                author = source_doc.metadata.get("author", "Unknown")
                year = extract_year(source_doc.metadata.get('creationdate', ''))
                page = int(source_doc.metadata.get("page", 0)) + 1

                # if not verify_quote_in_source(c.quote, formatted_docs):
                #     print('skipping')
                #     continue

                citations_str += (
                    f'Quote {idx}: "{c.quote}" ({author}, {year}, p. {page})\n\n'
                )
            full_response = f"**Answer:** {quoted_answer.answer}\n\n**Citations:**\n{citations_str}"

        await cl.Message(content=full_response).send()
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()