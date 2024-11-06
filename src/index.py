import os
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from langchain_openai import OpenAI  # 使用 langchain_openai 初始化 LLM
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 設定 LINE 和 OpenAI 的 API 金鑰
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
GPT_API_KEY = os.getenv("OPENAI_API_KEY")

# 驗證環境變數
if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_CHANNEL_SECRET or not GPT_API_KEY:
    raise ValueError("API keys for LINE or OpenAI are missing.")

# 初始化 FastAPI 應用
app = FastAPI()

# 初始化 LINE Bot API 和 Webhook Handler
try:
    line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
    handler = WebhookHandler(LINE_CHANNEL_SECRET)
except LineBotApiError as e:
    raise HTTPException(status_code=500, detail=f"LINE API initialization error: {e}")

# 初始化 OpenAI 模型
llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0,
    max_retries=2,
    api_key=GPT_API_KEY
)

@app.post("/callback")
async def callback(request: Request):
    # 取得 X-Line-Signature 標頭
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()

    # 驗證簽名
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
async def handle_message(event: MessageEvent):
    # 使用 LangChain 的 OpenAI 模型來生成回應
    user_message = event.message.text
    response = llm(user_message)

    # 將 GPT 的回應發送給用戶
    try:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response)
        )
    except LineBotApiError as e:
        raise HTTPException(status_code=500, detail=f"LINE Bot reply error: {e}")

# 運行應用程式
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
