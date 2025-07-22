
import os
import glob
import jnius_config

# Thiết lập classpath cho JVM trước khi import bất kỳ gì dùng jnius....
vncorenlp_dir = "/opt/chatbot_env/vncorenlp"
jar_path = os.path.join(vncorenlp_dir, "VnCoreNLP-1.2.jar")
libs_jars = glob.glob(os.path.join(vncorenlp_dir, "libs", "*.jar"))
jnius_config.set_classpath(jar_path, *libs_jars)

from py_vncorenlp import VnCoreNLP
from phonlp import download as phonlp_download, load as phonlp_load
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import oracledb
from typing import List

# Khởi tạo FastAPI
app = FastAPI(title="ChatGPT-style QA for Oracle", version="1.0")

# Thêm middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # Cho phép mọi origin
    allow_credentials=True,
    allow_methods=["*"],           # Cho phép tất cả phương thức (POST, GET, OPTIONS…)
    allow_headers=["*"],           # Cho phép mọi header
)

# Cấu hình Oracle DSN & tài khoản
ORACLE_USER = "tlu"
ORACLE_PWD = "tlu"
ORACLE_DSN = "45.122.253.178:2151/cdb2"  # host:port/service

# Kết nối Oracle
connection = oracledb.connect(user=ORACLE_USER, password=ORACLE_PWD, dsn=ORACLE_DSN)

# Tải mô hình PhoNLP pre-trained (1 lần)
MODEL_DIR = '/opt/chatbot_env/pretrained_phonlp'
if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
    phonlp_download(save_dir=MODEL_DIR)
nlp_model = phonlp_load(save_dir=MODEL_DIR)

# Khởi tạo bộ phân tách từ tiếng Việt (VnCoreNLP)
rdrsegmenter = VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)

# Khai báo model cho request/response
class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str

# API xử lý câu hỏi
@app.post("/api/chat", response_model=Answer)
def answer_question(q: Question):
    user_question = q.question

    # Bước 1: Tách từ => danh sách token
    tokens: List[str] = rdrsegmenter.word_segment(user_question)

    # Bước 2: Nối thành chuỗi cách nhau bởi dấu space
    text_for_annotate = " ".join(tokens)

    # Bước 3: Phân tích ngôn ngữ với PhoNLP (bây giờ đúng kiểu str)
    analysis = nlp_model.annotate(text=text_for_annotate)

    # Bước 4: Sinh câu SQL từ câu hỏi
    sql_query = generate_sql_from_question(user_question, analysis)

    # Bước 5: Thực thi SQL và định dạng kết quả
    result_text = execute_sql_and_format(sql_query)

    return Answer(answer=result_text)


def generate_sql_from_question(question_text, analysis):
    text = question_text.lower()
    if "bao nhiêu" in text and "nhân viên" in text:
        return "SELECT COUNT(*) FROM nhan_vien"
    return "SELECT 'Chua ho tro' FROM dual"

def execute_sql_and_format(sql):
    cur = connection.cursor()
    try:
        cur.execute(sql)
    except Exception as e:
        return f"Lỗi khi thực thi SQL: {e}"
    rows = cur.fetchall()
    if not rows:
        return "Không tìm thấy dữ liệu phù hợp."
    if len(rows) == 1 and len(rows[0]) == 1:
        return f"Kết quả là {rows[0][0]}."
    result_lines = []
    col_names = [d[0] for d in cur.description]
    for r in rows:
        row_str = ", ".join(f"{col_names[i]}: {r[i]}" for i in range(len(r)))
        result_lines.append(row_str)
    return ";\n".join(result_lines)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
