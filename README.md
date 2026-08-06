# RAG 视频脚本检索系统

  ## 项目目标

  为视频创作者提供一个基于 RAG 的素材检索和脚本生成系统。
  输入创作主题 → 检索相关脚本片段 → 生成脚本初稿。

  ## 技术栈

  - 检索：LlamaIndex + Chroma（稠密向量）+ BM25（稀疏检索）
  - 融合：RRF（Reciprocal Rank Fusion）
  - 重排序：Cross-Encoder ReRank
  - 生成：DeepSeek-R1-7B（Ollama 本地推理）
  - API：FastAPI
  - 部署：Docker Compose
  - 前端：Gradio

  ## 架构
  
  ```                     
  用户 → Gradio 前端 → FastAPI → 检索器 → 生成器 → 返回       
                          ↓  
                      Chroma 向量库
  ```

  ## 进度

  - [ ] 第1天：LangChain + Chroma 最简 RAG 链路
  - [ ] 第2天：对比 3 种分块策略
  - [ ] 第3天：BM25 + 稠密混合检索 + RRF
  - [ ] 第4天：FastAPI 包装
  - [ ] 第5天：Docker Compose
  - [ ] 第6天：Gradio 前端 + GIF demo
