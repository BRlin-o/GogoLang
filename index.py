from fastapi import FastAPI, Request
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from langchain import OpenAI
from langchain.chains import ConversationChain
import os

# 設置環境變數
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GPT_API_KEY = os.getenv("GPT_API_KEY")

# 初始化 FastAPI 應用
app = FastAPI()

# 初始化 LINE Bot API 和 Webhook Handler
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 初始化 LangChain Conversation Chain 和 OpenAI GPT-4 mini
llm = OpenAI(api_key=GPT_API_KEY, model="gpt-4-mini")
conversation = ConversationChain(llm=llm)

# 路由設定，接收 LINE Webhook 請求
@app.post("/callback")
async def callback(request: Request):
    # 取得簽名以驗證請求合法性
    signature = request.headers["X-Line-Signature"]
    body = await request.body()

    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        return "Invalid signature. Please check your channel access token/channel secret."

    return "OK"

# 處理收到的文字訊息事件
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text  # 使用者發送的訊息

    # 使用 LangChain Conversation Chain 生成回應
    response = conversation.run(input=user_message)

    # 回傳生成的回應至 LINE 使用者
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response)
    )

# 運行應用程式
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
