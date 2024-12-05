from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.prompts.prompt import PromptTemplate
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever

from src.constants import SCOOTER_NAMES

def create_gogoro_tool(kb_id, llm):
    """
    構建檢索工具，專用於 GOGORO 車輛手冊查詢。
    """
    # 定義 Prompt 模板
    rag_prompt_template = """只能使用以下內容來回答最後的問題，並遵守以下規則：
1. 如果不知道答案，請不要編造答案。只需說 **找不到最終答案**。
2. 如果找到答案，請務必詳細地正確回答，並附上**直接**用來得出答案的來源列表。排除與最終答案無關的來源。
3. 如果score最高的內容包含 Markdown 格式的 ![]()，請在回應中保留它完整的樣貌（一模一樣）。
4. 完全使用繁體中文進行溝通。

{context}

問題：{question}
有幫助的回答："""
    rag_prompt = PromptTemplate(template=rag_prompt_template)

    # 定義工具的檢索函數
    def retrieval_func(question: str, scooter_name: str) -> str:
        """
        執行檢索，基於問題與車款名稱，並處理潛在的檢索錯誤。
        """
        try:
            # 初始化檢索器
            retriever = AmazonKnowledgeBasesRetriever(
                knowledge_base_id=kb_id,
                retrieval_config={
                    "vectorSearchConfiguration": {
                        "numberOfResults": 5,
                        "filter": {
                            "equals": {"key": "scooter_name", "value": scooter_name},
                        }
                    }
                },
            )
            # 初始化檢索鏈
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                input_key="question",
                prompt=rag_prompt,
                verbose=True,
            )
            # 執行檢索
            answer = rag_chain.run({"question": question})
            print("[DEUBG] RAG answer:", answer)
            return answer
        
        except ConnectionError:
            return "無法連接到知識庫服務，請稍後再試。"
        except Exception as e:
            # 捕捉所有其他異常，並返回簡單錯誤訊息
            return f"檢索過程中發生錯誤：{str(e)}"

    # 更新工具描述
    gogoro_tool_description = (
        "此工具用於查詢 GOGORO 車輛手冊，提供詳細操作指引或技術信息。"
        "需要提供車款名稱（例如 'S3'）和具體問題描述，工具將基於資料庫進行檢索並返回答案。"
        "支持的車款名稱僅包括：\n" + ", ".join(SCOOTER_NAMES) + "。\n"
        "注意：若連線問題或數據庫無匹配資料，工具將提示錯誤訊息或返回找不到結果。"
    )

    gogoro_tool = Tool(
        name="GogoroManualSearch",
        description=gogoro_tool_description,
        func=retrieval_func,
        # return_direct=True  # Tool 呼叫後直接返回答案
    )
    return gogoro_tool
