from langchain.chains import RetrievalQA
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

rag_prompt_template = """只能使用以下內容來回答最後的問題，並遵守以下規則：
1. 如果不知道答案，請不要編造答案。只需說 **找不到最終答案**。
2. 如果找到答案，請務必詳細地正確回答，並附上**直接**用來得出答案的來源列表。排除與最終答案無關的來源。
3. 如果score最高的內容包含 Markdown 格式的 ![]()，請在回應中保留它完整的樣貌（一模一樣）。
4. 完全使用繁體中文進行溝通。

{context}

問題：{question}
有幫助的回答："""

RAG_PROMPT = PromptTemplate.from_template(template=rag_prompt_template)

def get_rag_chain(kb_id, llm, car_model=None):
    # _car_model = car_model.replace("CrossOver", "Crossover").replace("VIVA", "Viva").replace("XL", "Xl").replace("MIX", "Mix").replace("JEGO", "Jego").replace("S1", "S1 / 1").replace("S2", "S2 / 2").replace("S3", "S3 / 3")
    # _car_model = car_model.replace("VIVA", "Viva").replace("XL", "Xl").replace("MIX", "Mix").replace("JEGO", "Jego").replace("S1", "S1 / 1").replace("S2", "S2 / 2").replace("S3", "S3 / 3")
    # _car_model = f"Gogoro {_car_model}"
    print("[DEBUG] car_model", car_model)

    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=kb_id,
        retrieval_config={
            "vectorSearchConfiguration": {
                "numberOfResults": 5,
                "filter": {
                    "equals": { "key": "car_model", "value": car_model.upper() }
                    # "equals": { "key": "scooter_name", "value": car_model }
                }
            }
        },
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        # chain_type_kwargs={"prompt": RAG_PROMPT},
        input_key="question",
    )