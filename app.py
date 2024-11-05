import os
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from langchain import OpenAI
from langchain.chains import ConversationChain

# 初始化 Flask 應用
app = Flask(__name__)

# 設定 Line Bot 的 Channel Secret 和 Access Token
channel_secret = os.getenv("CHANNEL_SECRET")
channel_access_token = os.getenv("CHANNEL_ACCESS_TOKEN")
gpt_api_key = os.getenv("GPT_API_KEY")

line_bot_api = LineBotApi(channel_access_token)
handler = WebhookHandler(channel_secret)

# 初始化 LangChain Conversation Chain 和 OpenAI GPT-4 mini
llm = OpenAI(api_key=gpt_api_key, model="gpt-4-mini")
conversation = ConversationChain(llm=llm)

@app.route("/callback", methods=["POST"])
def callback():
    # 取得 Line 請求的簽名，驗證其合法性
    signature = request.headers["X-Line-Signature"]
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return "OK"

# 處理文字訊息事件
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text

    # 使用 LangChain Conversation Chain 生成回應
    response = conversation.run(input=user_message)

    # 回傳生成的回應至 LINE 使用者
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response)
    )

if __name__ == "__main__":
    app.run()
