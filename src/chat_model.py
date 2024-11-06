from langchain.schema import HumanMessage

class ChatModel:
    def __init__(self, llm):
        self.llm = llm
        self.conversation = []

    def get_response(self, user_input):
        # 新增用戶訊息至對話記錄
        self.conversation.append(HumanMessage(content=user_input))
        
        # 取得 LLM 回應
        response = self.llm(self.conversation)
        
        # 新增 AI 回應至對話記錄
        self.conversation.append(response)
        
        print("AI回答內容：")
        print(response.content.strip())
        
        return response.content.strip()
