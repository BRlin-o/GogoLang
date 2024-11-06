import uvicorn
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

from line_basic import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
