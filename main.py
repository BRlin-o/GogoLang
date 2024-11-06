import uvicorn
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# from api.line_basic import app

from api.index import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)