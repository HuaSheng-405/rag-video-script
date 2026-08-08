from llama_index.core import SimpleDirectoryReader,VectorStoreIndex,Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core.retrievers import VectorIndexRetriever,QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
import chromadb
import requests
import jieba

# ========== 配置 ==========
Settings.embed_model = HuggingFaceEmbedding(
    model_name="model/multilingual-MiniLM"
)
Settings.llm = None
Settings.text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=50)

# ========== 加载文档 ==========
documents = SimpleDirectoryReader("data/scripts").load_data()
print(f"加载文档: {len(documents)} 篇")
nodes = Settings.text_splitter.get_nodes_from_documents(documents)
# ========== 建稠密向量索引 ==========
chroma_client = chromadb.PersistentClient(path = "chromadb")
collection = chroma_client.get_or_create_collection("video_scripts")
existing = collection.get()
if existing["ids"]:
    collection.delete(ids = existing['ids'])

vector_store = ChromaVectorStore(chroma_collection = collection)
storage_context = StorageContext.from_defaults(vector_store = vector_store)
index = VectorStoreIndex(nodes = nodes,storage_context = storage_context)
dense_retriever = index.as_retriever(similarity_top_k = 5)

# ========== 建 BM25 稀疏检索器 ==========
for node in nodes:
    node.set_content(" ".join(jieba.cut(node.text)))
bm25_retriever = BM25Retriever.from_defaults(nodes = nodes,similarity_top_k = 5)

# ========== 混合检索：RRF 融合 ==========
hybrid_retriever = QueryFusionRetriever(
    retrievers = [dense_retriever,bm25_retriever],
    similarity_top_k = 5,
    mode = "reciprocal_rerank",
    num_queries = 1,
)

# ========== 对比三种检索 ==========
queries = [
    "推荐一个美食店铺",
    "iPhone 17 的使用体验",
    "秦始皇统一文字的历史意义",
    "如何提高学习效率",
    "智能戒指的功能介绍",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"Query: {query}")

    # 稠密向量
    dense_results = dense_retriever.retrieve(query)
    print(f"\n【稠密向量 Top-3】")
    for i,n in enumerate(dense_results[:3]):
        source = n.node.metadata.get("file_name","?")
        print(f"    {i + 1}.score={n.score:.4f} source={source} text={n.text[:60]}...")

    # BM25
    bm25_results = bm25_retriever.retrieve(query)
    print(f"\n【BM25 Top-3】")
    for i,n in enumerate(bm25_results[:3]):
        source = n.node.metadata.get("file_name","?")
        print(f"    {i + 1}.score={n.score:.4f} source={source} text={n.text[:60]}...")

    # 混合(RRF)
    hybrid_results = hybrid_retriever.retrieve(query)
    print(f"\n【混合检索 RRF Top-3】")
    for i,n in enumerate(hybrid_results[:3]):
        source = n.node.metadata.get("file_name","?")
        print(f"    {i + 1}.score={n.score:.4f} source={source} text={n.text[:60]}...")