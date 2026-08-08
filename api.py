from fastapi import FastAPI,UploadFile,File
from pydantic import BaseModel
import shutil
import os
from rag_engine import build_index,rag_query

# 启动时建一次索引
print("正在加载索引...")
retriever = build_index()
print("索引就绪")

app = FastAPI(title = "RAG 视频脚本检索 API")

class QueryRequest(BaseModel):
    question:str

class QueryResponse(BaseModel):
    answer:str
    sources:list[str]

@app.post("/upload")
async def upload_script(file:UploadFile = File(...)):
    """上传新的视频脚本素材"""
    os.makedirs("data/scripts",exist_ok = True)
    path = f"data/scripts/{file.filename}"
    with open(path,"wb") as f:
        shutil.copyfileobj(file.file,f)
    return {"filename":file.filename,"status":"上传成功"}

@app.post("/query",response_model = QueryResponse)
def query_scripts(req:QueryRequest):
    """检索+生成回答"""
    result = rag_query(req.question,retriever)
    return QueryResponse(answer=result["answer"],sources=result["sources"])

@app.get("/health")
def health():
    return {"status":"ok"}