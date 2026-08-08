from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
from openai import OpenAI
import os

# ========== 全局配置 ==========
Settings.embed_model = HuggingFaceEmbedding(
    model_name = "model\multilingual-MiniLM"
)
Settings.text_splitter = SentenceSplitter(chunk_size = 256,chunk_overlap = 50)
client = OpenAI(base_url = "http://localhost:11434/v1",api_key = "ollama")

def build_index(data_dir = "data\scripts",db_dir = "chroma_db",collection_name = "video_scripts"):
    """加载文档、建索引,返回retriever"""
    documents = SimpleDirectoryReader(data_dir).load_data()
    print(f"加载文档:{len(documents)}篇")
    nodes = Settings.text_splitter.get_nodes_from_documents(documents)
    chroma_client = chromadb.PersistentClient(path = "chroma_db")
    collection = chroma_client.get_or_create_collection(collection_name)

    # 清空旧数据
    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids = existing)
        print(f"已清空{len(existing)}条旧数据")

    vector_store = ChromaVectorStore(chroma_collection = collection)
    storage_context = StorageContext.from_defaults(vector_store = vector_store)
    index = VectorStoreIndex(nodes = nodes,storage_context = storage_context)
    print("索引构建完成")

    return index.as_retriever(similarity_top_k = 3)

def rag_query(query,retriever,model = "deepseek-r1:1.5b"):
    """检索 + 生成回答，返回 (answer, sources)"""
    nodes = retriever.retrieve(query)
    context = "\n\n".join([n.text for n in nodes])

    prompt = f"""
    根据以下素材回答问题。只使用素材中提到的信息，不要编造。

    素材：
    {context}

    用户：{query}
    助手：

    """
    response = client.chat.completions.create(
        model = model,
        messages = [{"role":"user","content":prompt}],
        temperature = 0.7,
    )
    sources = [n.node.metadata.get("file_name","?") for n in nodes]

    return {
        "answer":response.choices[0].message.content,
        "sources":sources,
        "context":context,
    }

if __name__ == "__main__":
    retriever = build_index()

    print("\n" + "=" * 60)
    print("RAG 测试")
    print("=" * 60)

    result = rag_query("推荐一个美食店铺", retriever)
    print(f"回答: {result['answer']}")
    print(f"来源: {result['sources']}")