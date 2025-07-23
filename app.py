
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import oracledb
import os
import glob
import jnius_config

# Chỉ dùng alias_mapping, bỏ MT5
from alias_mapping import alias_mapping

# Thiết lập classpath cho JVM
vncorenlp_dir = "/opt/chatbot_env/vncorenlp"
jar_path = os.path.join(vncorenlp_dir, "VnCoreNLP-1.2.jar")
libs_jars = glob.glob(os.path.join(vncorenlp_dir, "libs", "*.jar"))
jnius_config.set_classpath(jar_path, *libs_jars)

from py_vncorenlp import VnCoreNLP
from phonlp import download as phonlp_download, load as phonlp_load
import uvicorn

# Tạo FastAPI app
app = FastAPI(title="ChatGPT-style QA for Oracle (No MT5)", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối Oracle
ORACLE_USER = "spp"
ORACLE_PWD = "Ab123456"
ORACLE_DSN = "45.122.253.178:2151/cdb2"
try:
    connection = oracledb.connect(user=ORACLE_USER, password=ORACLE_PWD, dsn=ORACLE_DSN)
except Exception as e:
    print("Lỗi kết nối Oracle:", e)
    connection = None

# Tải PhoNLP
MODEL_DIR = "/opt/chatbot_env/pretrained_phonlp"
if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
    phonlp_download(save_dir=MODEL_DIR)

try:
    nlp_model = phonlp_load(save_dir=MODEL_DIR)
except Exception as e:
    print(f"Lỗi tải PhoNLP: {e}")
    nlp_model = None

# Tải VnCoreNLP
try:
    rdrsegmenter = VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)
except Exception as e:
    print(f"Lỗi khởi tạo VnCoreNLP: {e}")
    rdrsegmenter = None

# Định nghĩa schema API
class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str


@app.post("/api/chat", response_model=Answer)
def answer_question(q: Question):
    user_question = q.question.lower()

    # Check alias mapping
    for keyword in alias_mapping:
        if keyword in user_question:
            mapped_sql = alias_mapping[keyword]

            # Lấy phần còn lại làm giá trị
            value = user_question.split(keyword, 1)[-1].strip().strip("?.,")
            sql_query = mapped_sql.replace("{value}", value)

            print(f"[INFO] SQL sinh ra từ alias_mapping: {sql_query}")
            result_text = execute_sql_and_format(sql_query)
            return Answer(answer=result_text)
    return Answer(answer="Không hiểu câu hỏi hoặc chưa được hỗ trợ.")

def execute_sql_and_format(sql: str) -> str:
    if not connection:
        return "Lỗi kết nối CSDL."

    cur = connection.cursor()
    try:
        print("SQL:", sql)
        cur.execute(sql)
    except Exception as e:
        return f"Lỗi khi thực thi SQL: {e}"

    rows = cur.fetchall()
    if not rows:
        return "Không tìm thấy dữ liệu phù hợp."

    if len(rows) == 1 and len(rows[0]) == 1:
        return f"Kết quả là {rows[0][0]}."

    import json

    col_names = [d[0] for d in cur.description]

    # Hiển thị bảng markdown nếu ít hơn 20 dòng
    if len(rows) <= 20:
        header = "| " + " | ".join(col_names) + " |"
        divider = "| " + " | ".join(["---"] * len(col_names)) + " |"
        table_rows = []
        for row in rows:
            row_values = [str(cell).replace("|", "\\|") for cell in row]  # escape dấu | nếu có
            table_rows.append("| " + " | ".join(row_values) + " |")
        return "\n".join([header, divider] + table_rows)

    data = [dict(zip(col_names, row)) for row in rows]
    return f"```json\n{json.dumps(data, indent=2, ensure_ascii=False)}\n```"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
