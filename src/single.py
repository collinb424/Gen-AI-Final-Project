from typing import TypedDict
from llm import LLM
from langchain.prompts import PromptTemplate
from vectorstore import VECTOR_STORE
from langgraph.graph import StateGraph, START
from langchain.schema.runnable import Runnable


# RAG
PROMPT = """
You're an expert in answering questions. Use the following pieces of documents relevant to the question to answer it. Make sure to not make up new information and only use the information provided in the documents. Always add citations to your answer.
Question: {question}
Context: {context}
Answer:
""".strip()

PROMPT = PromptTemplate.from_template(PROMPT)

class State(TypedDict):
    question: str
    context: list
    answer: str
    

def retrieve(state):
    results = VECTOR_STORE.similarity_search(state["question"], k=5)
    return {"context": results}

def generate(state):
    doc_text = "\n\n".join(doc.page_content for doc in state["context"])
    msgs = PROMPT.invoke({"question": state["question"], "context": doc_text})
    response = LLM.invoke(msgs)
    return {"answer": response.content}



# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

q = input("Enter a question: ")
response = graph.invoke({"question": q})
print(response["answer"])