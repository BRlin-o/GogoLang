from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from GoLang.src.llms.llm_openai import Chat_OpenAI
from GoLang.src.chat_model import ChatModel
import os

app = FastAPI()
line_bot_api = LineBotApi(os.getenv("LINE_CHANNEL_ACCESS_TOKEN"))
handler = WebhookHandler(os.getenv("LINE_CHANNEL_SECRET"))

# 建立 ChatModel 實例
llm = Chat_OpenAI()
chat_model = ChatModel(llm=llm)

@app.get("/")
async def root():
    return {"message": "I'm healthy :D"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers["X-Line-Signature"]
    body = await request.body()
    try:
        handler.handle(body.decode(), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Missing Parameters")
    return "OK"

@handler.add(MessageEvent, message=TextMessage)
def handling_message(event):
    if isinstance(event.message, TextMessage):
        user_message = event.message.text
        reply_msg = chat_model.get_response(user_message)
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply_msg))