from llama_index.core import SimpleDirectoryReader,Settings
from llama_index.core.node_parser import SentenceSplitter,TokenTextSplitter,SimpleFileNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex,StorageContext
import chromadb
from llama_index.core.ingestion import IngestionPipeline

# ========== 配置 ==========
Settings.embed_model = HuggingFaceEmbedding(
    model_name = 'model/multilingual-MiniLM'
)

documents = SimpleDirectoryReader('data/scripts').load_data()
print(f'加载文档：{len(documents)}篇')

# ========== 测试函数 ==========
def test_chunker(name,splitter,documents):
    chroma_client = chromadb.PersistentClient(path = 'chroma_db')
    collection = chroma_client.get_or_create_collection(f"chunk_test_{name}")
    existing = collection.get()
    if existing['ids']:
        collection.delete(ids = existing["ids"])

    vector_store = ChromaVectorStore(chroma_collection = collection)
    storage_context = StorageContext.from_defaults(vector_store = vector_store)
    if splitter:
        pipeline = IngestionPipeline(transformations = [splitter,Settings.embed_model])
        nodes = pipeline.run(documents = documents)
    else:
        nodes = documents
    index = VectorStoreIndex(nodes = nodes,storage_context = storage_context,transformations=[splitter])

    n_chunks = len(collection.get()['ids'])
    print(f"\n{name}:切出{n_chunks}个chunk")

    # 用相同的query检索 看结果
    retriever = index.as_retriever(similarity_top_k = 3)
    queries = ["美食探店","iphone体验","历史知识"]
    for q in queries:
        nodes = retriever.retrieve(q)
        print(f"    [{q}] score=[{nodes[0].score:.4f}]:{nodes[0].text[:]}")

# ========== 三种策略 ==========
if __name__ == '__main__':
    # 1.不切分（整篇文档作为一个 chunk）
    test_chunker("None",None,documents)

    # 2.按句子切分（SentenceSplitter）
    test_chunker("SentenceSplitter",SentenceSplitter(chunk_size = 256,chunk_overlap = 50),documents)

    # 3.按 token 数硬切（TokenTextSplitter）
    test_chunker("TokenTextSplitter",TokenTextSplitter(chunk_size = 256,chunk_overlap = 50),documents)
