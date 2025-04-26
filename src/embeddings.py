from langchain_openai import OpenAIEmbeddings

_MODEL = "text-embedding-3-large"

EMBEDDINGS = OpenAIEmbeddings(model=_MODEL, api_key=_API_KEY)