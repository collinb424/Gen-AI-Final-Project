from embeddings import EMBEDDINGS
from langchain_milvus import Milvus

from pymilvus import Collection, MilvusException, connections, db, utility

MILVUS_URI = "http://milvus:19530"
DATABASE_NAME = "assignment_rag"
COLLECTION_NAME = "assignment_rag"

connections.connect(
    host=MILVUS_URI.split("//")[1].split(":")[0],
    port=int(MILVUS_URI.split("//")[1].split(":")[1])
)
_existing_databases = db.list_database()
if DATABASE_NAME not in _existing_databases:
    db.create_database(DATABASE_NAME)
db.using_database(DATABASE_NAME)

def drop_collection():
    collections = utility.list_collections()
    if COLLECTION_NAME in collections:
        col = Collection(name=COLLECTION_NAME)
        col.drop()

VECTOR_STORE = Milvus(
    embedding_function=EMBEDDINGS,
    connection_args={"uri": MILVUS_URI, "db_name": "assignment_rag"},
    index_params={"index_type": "FLAT", "metric_type": "COSINE"},
    collection_name=COLLECTION_NAME,
    auto_id=True,
) 