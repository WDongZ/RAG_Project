# RAG 知识问答系统

这是一个基于 LangChain 和 Streamlit 构建的检索增强生成（RAG）知识问答系统。用户可以上传文档，系统会基于这些文档回答用户的问题。

## 功能特性

- 📁 **多格式文档支持**：支持 .txt 和 .md 格式的文档上传
- 🔍 **智能检索**：使用向量相似度搜索相关文档片段
- 💬 **对话界面**：友好的聊天式交互界面
- 📊 **实时状态**：显示系统运行状态和对话统计
- 🔄 **灵活管理**：支持清空对话历史和重置系统

## 系统架构

```
用户上传文档 → 文档分割 → 向量化 → 存储到向量数据库
                                            ↓
用户提问 → 检索相关片段 → 生成上下文 → LLM生成回答
```

## 技术栈

- **前端框架**：Streamlit
- **RAG框架**：LangChain
- **向量数据库**：Chroma
- **嵌入模型**：sentence-transformers/all-MiniLM-L6-v2
- **语言模型**：Ollama (Llama2)

## 安装和运行

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd RAG_Project
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装和启动 Ollama

首先安装 Ollama：

```bash
# Linux/macOS
curl -fsSL https://ollama.ai/install.sh | sh

# 或者访问 https://ollama.ai 下载对应系统的版本
```

启动 Ollama 服务并下载模型：

```bash
# 启动 Ollama 服务
ollama serve

# 在另一个终端下载 Llama2 模型
ollama pull llama2
```

### 4. 运行应用

```bash
streamlit run streamlit_app.py
```

应用将在浏览器中打开，默认地址为 `http://localhost:8501`

## 使用方法

1. **上传文档**：
   - 在左侧边栏点击"选择文档文件"
   - 上传 .txt 或 .md 格式的文档
   - 点击"处理文档"按钮

2. **开始对话**：
   - 在主界面的文本输入框中输入问题
   - 点击"提问"按钮获取答案

3. **管理对话**：
   - 查看右侧的系统状态
   - 使用"清空对话历史"重新开始
   - 使用"重置系统"完全重置

## 配置选项

### 文本分割参数

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # 每个文档片段的大小
    chunk_overlap=50     # 片段之间的重叠字符数
)
```

### 检索参数

```python
retriever = vector_store.as_retriever(
    search_type="similarity",  # 搜索类型
    search_kwargs={"k": 3}     # 返回最相关的3个片段
)
```

## 自定义和扩展

### 添加新的文档格式

在 `load_documents` 函数中添加新的文件类型支持：

```python
if uploaded_file.name.endswith('.pdf'):
    from langchain.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    documents.extend(loader.load())
```

### 使用不同的语言模型

修改 `create_qa_chain` 函数中的 LLM 配置：

```python
# 使用 OpenAI GPT
from langchain.llms import OpenAI
llm = OpenAI(temperature=0.7)

# 使用 HuggingFace 模型
from langchain.llms import HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="gpt2",
    task="text-generation"
)
```

### 自定义提示模板

修改 `prompt_template` 变量以改变系统的回答风格：

```python
prompt_template = """
你是一个专业的AI助手。基于以下上下文信息，用中文详细回答用户的问题。

上下文：{context}
问题：{question}

请提供准确、有用的回答：
"""
```

## 故障排除

### 常见问题

1. **Ollama 连接失败**
   - 确保 Ollama 服务正在运行：`ollama serve`
   - 检查模型是否已下载：`ollama list`

2. **文档加载失败**
   - 检查文件编码是否为 UTF-8
   - 确保文件格式受支持

3. **向量数据库错误**
   - 检查磁盘空间是否足够
   - 尝试重置系统后重新处理文档

### 性能优化

- 对于大型文档，考虑增加 `chunk_size` 参数
- 使用更强大的嵌入模型提高检索质量
- 调整检索参数 `k` 值平衡相关性和响应速度

## 贡献指南

欢迎提交 Issue 和 Pull Request 来改进这个项目！

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements.txt
pip install black flake8 pytest

# 代码格式化
black streamlit_app.py

# 代码检查
flake8 streamlit_app.py
```

## 许可证

MIT License

## 更新日志

### v1.0.0
- 初始版本发布
- 支持 .txt 和 .md 文档上传
- 集成 Ollama Llama2 模型
- 实现基础 RAG 功能

---

如有问题或建议，请联系项目维护者或提交 Issue。
