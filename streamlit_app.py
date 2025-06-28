import streamlit as st
import os
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import tempfile

# 设置页面配置
st.set_page_config(
    page_title="RAG 知识问答系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化 session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def load_documents(uploaded_files):
    """加载上传的文档"""
    documents = []
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            # 保存上传的文件到临时目录
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # 根据文件类型加载文档
            if uploaded_file.name.endswith(('.txt', '.md')):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
    
    return documents

def create_vector_store(documents):
    """创建向量数据库"""
    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    
    # 创建嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # 创建向量数据库
    vector_store = Chroma.from_documents(
        documents=texts,
        embedding=embeddings
    )
    
    return vector_store

def create_qa_chain(vector_store, llm_type="ollama"):
    """创建问答链"""
    
    # 创建检索器
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # 创建提示模板
    prompt_template = """
    基于以下上下文信息，请回答用户的问题。如果上下文中没有相关信息，请诚实地说"我不知道"。

    上下文信息：
    {context}

    问题：{question}

    回答：
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # 选择语言模型
    if llm_type == "ollama":
        try:
            llm = Ollama(model="llama2")
        except:
            st.error("无法连接到 Ollama，请确保 Ollama 服务正在运行")
            return None
    else:
        st.error("暂时只支持 Ollama 模型")
        return None
    
    # 创建问答链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

def main():
    st.title("🤖 RAG 知识问答系统")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("📁 文档上传")
        uploaded_files = st.file_uploader(
            "选择文档文件",
            type=['txt', 'md'],
            accept_multiple_files=True,
            help="支持 .txt 和 .md 格式的文件"
        )
        
        if uploaded_files:
            st.success(f"已上传 {len(uploaded_files)} 个文件")
            
            if st.button("🔄 处理文档", type="primary"):
                with st.spinner("正在处理文档..."):
                    try:
                        # 加载文档
                        documents = load_documents(uploaded_files)
                        st.success(f"成功加载 {len(documents)} 个文档")
                        
                        # 创建向量数据库
                        vector_store = create_vector_store(documents)
                        st.session_state.vector_store = vector_store
                        st.success("向量数据库创建成功！")
                        
                        # 创建问答链
                        qa_chain = create_qa_chain(vector_store)
                        if qa_chain:
                            st.session_state.qa_chain = qa_chain
                            st.success("问答系统初始化成功！")
                        
                    except Exception as e:
                        st.error(f"处理文档时出错：{str(e)}")
        
        st.markdown("---")
        
        # 系统设置
        st.header("⚙️ 系统设置")
        
        # 清空对话历史
        if st.button("🗑️ 清空对话历史"):
            st.session_state.chat_history = []
            st.success("对话历史已清空")
        
        # 重置系统
        if st.button("🔄 重置系统"):
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.session_state.chat_history = []
            st.success("系统已重置")
    
    # 主界面
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 对话界面")
        
        # 显示对话历史
        chat_container = st.container()
        with chat_container:
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                st.markdown(f"**🙋‍♂️ 用户：** {question}")
                st.markdown(f"**🤖 助手：** {answer}")
                st.markdown("---")
        
        # 问题输入
        if st.session_state.qa_chain:
            question = st.text_input(
                "请输入您的问题：",
                placeholder="在这里输入您想问的问题...",
                key="question_input"
            )
            
            col_ask, col_clear = st.columns([3, 1])
            
            with col_ask:
                if st.button("🚀 提问", type="primary", use_container_width=True):
                    if question:
                        with st.spinner("正在思考..."):
                            try:
                                result = st.session_state.qa_chain({"query": question})
                                answer = result['result']
                                
                                # 添加到对话历史
                                st.session_state.chat_history.append((question, answer))
                                
                                # 重新运行以更新界面
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"回答问题时出错：{str(e)}")
                    else:
                        st.warning("请输入问题")
        else:
            st.info("请先上传文档并处理后再开始提问")
    
    with col2:
        st.header("📊 系统状态")
        
        # 显示系统状态
        if st.session_state.vector_store:
            st.success("✅ 向量数据库已就绪")
        else:
            st.warning("⏳ 向量数据库未初始化")
        
        if st.session_state.qa_chain:
            st.success("✅ 问答系统已就绪")
        else:
            st.warning("⏳ 问答系统未初始化")
        
        st.markdown(f"**对话轮数：** {len(st.session_state.chat_history)}")
        
        # 使用说明
        st.markdown("---")
        st.header("📝 使用说明")
        st.markdown("""
        1. **上传文档**：在左侧上传 .txt 或 .md 格式的文档
        2. **处理文档**：点击"处理文档"按钮创建知识库
        3. **开始提问**：在对话界面输入问题并提问
        4. **查看回答**：系统会基于上传的文档回答问题
        
        **注意事项：**
        - 确保 Ollama 服务正在运行
        - 支持多文档同时上传
        - 可以随时清空对话历史或重置系统
        """)

if __name__ == "__main__":
    main()
