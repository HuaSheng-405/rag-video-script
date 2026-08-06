from llama_index.core import VectorStoreIndex,SimpleDirectoryReader,Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb
from openai import OpenAI
import os

os.makedirs("data/scripts", exist_ok=True)

# ========== 1. 配置 ==================================
# Embedding 模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name = "model/multilingual-MiniLM"
)
Settings.text_splitter = SentenceSplitter(chunk_size = 256,chunk_overlap = 50)

# Ollama LLM
client = OpenAI(base_url = "http://localhost:11434/v1",api_key = "ollama")

# ========== 2. 加载 + 切分 + 向量化 + 建索引 ==========
documents = SimpleDirectoryReader("data/scripts").load_data()
print(f"加载文档:{len(documents)}篇")

#Chroma
chroma_client = chromadb.PersistentClient(path = "chroma_db")
collection = chroma_client.get_or_create_collection("video_scripts")

#清空上次跑的旧数据
delete_data = collection.get()["ids"]
collection.delete(ids = collection.get()["ids"])
print(f"已清空{len(delete_data)}条旧数据")

vector_store = ChromaVectorStore(chroma_collection = collection)
storage_context = StorageContext.from_defaults(vector_store = vector_store)

index = VectorStoreIndex.from_documents(
    documents,storage_context = storage_context
)
print("索引构建完成")

# ========== 3. 检索 ===================================
retriever = index.as_retriever(similarity_top_k = 3)
nodes = retriever.retrieve("美食探店")
print("\n检索到{len(nodes)}条:")
for i,node in enumerate(nodes):
    print(f"    [{i + 1}] score={node.score:.4f}:{node.text[:80]}...")

# ========== 4. RAG 生成 ===============================
def rag_query(query):
    nodes = retriever.retrieve(query)
    context = "\n\n".join([n.text for n in nodes])

    prompt = f"""
    根据以下素材，用3句话向用户推荐一个美食店铺。只使用素材中提到的信息，不要编造。

    素材：
    {context}

    用户：{query}
    助手：

    """
    response = client.chat.completions.create(
        model = "deepseek-r1:1.5b",
        messages = [{"role":"user","content":prompt}],
        temperature = 0.7,
    )

    return response.choices[0].message.content

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("RAG 测试")
    print("=" * 60)
    answer = rag_query("推荐一个美食店铺")
    print(answer)