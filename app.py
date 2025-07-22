from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import os
import glob
import jnius_config

# Thiết lập classpath cho JVM
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
from alias_mapping import alias_mapping

# Load model MT5 đã fine-tune
model_dir = "./vncorenlp/models/vietext2sql_mt5"
t5_tokenizer = MT5Tokenizer.from_pretrained(model_dir)
t5_model = MT5ForConditionalGeneration.from_pretrained(model_dir)

# Tạo FastAPI app
app = FastAPI(title="ChatGPT-style QA for Oracle", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Kết nối Oracle
ORACLE_USER = "tlu"
ORACLE_PWD = "tlu"
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
    user_question = q.question

    if not rdrsegmenter or not nlp_model:
        return Answer(answer="Lỗi tải mô hình xử lý ngôn ngữ.")

    # Tách từ và phân tích
    tokens: List[str] = rdrsegmenter.word_segment(user_question)
    text_for_annotate = " ".join(tokens)
    _ = nlp_model.annotate(text=text_for_annotate)

    # Gợi nhớ từ khóa
    for keyword in alias_mapping:
        if keyword in user_question.lower():
            mapped_value = alias_mapping[keyword]
            print(f"Từ '{keyword}' ánh xạ tới: {mapped_value}")

    # Sinh câu SQL từ MT5
    sql_query = generate_sql_from_question(user_question)
    print(f"[INFO] SQL sinh ra: {sql_query}")

    # Thực thi SQL và trả kết quả
    result_text = execute_sql_and_format(sql_query)
    return Answer(answer=result_text)

def generate_sql_from_question(question_text: str) -> str:
    input_ids = t5_tokenizer.encode(question_text.lower(), return_tensors='pt')
    output_ids = t5_model.generate(input_ids, max_length=200, num_beams=4, early_stopping=True)
    sql = t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return sql

def execute_sql_and_format(sql: str) -> str:
    if not connection:
        return "Lỗi kết nối CSDL."

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

    col_names = [d[0] for d in cur.description]
    result_lines = []
    for r in rows:
        row_str = ", ".join(f"{col_names[i]}: {r[i]}" for i in range(len(r)))
        result_lines.append(row_str)

    return "\n".join(result_lines)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
