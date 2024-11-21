from fastapi import FastAPI, File, UploadFile
from io import StringIO
import os
import pandas as pd
import preprocessing  # preprocessing.py 모듈을 임포트

app = FastAPI()

# 파일 저장 위치 설정
UPLOAD_DIR = "uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

PROCESSED_DIR = "processed_files"
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

async def process_csv(file: UploadFile):
    """
    파일을 저장하고 전처리 후 처리된 파일을 반환하는 메서드
    """
    # 파일 이름 추출
    file_name = file.filename
    file_location = os.path.join(UPLOAD_DIR, file_name)
    
    # 파일을 메모리에서 읽고 Pandas로 처리하기
    contents = await file.read()
    string_io = StringIO(contents.decode())
    
    # pandas로 CSV 읽기
    data = pd.read_csv(string_io)
    
    # 데이터를 CSV 파일로 저장
    data.to_csv(file_location, index=False)
    
    # preprocessing.py의 main() 함수 호출하여 처리된 데이터 반환
    processed_data = preprocessing.main(file_location)  # 파일 경로를 전달
    
    # 저장된 처리된 파일 경로
    processed_file_location = os.path.join(PROCESSED_DIR, f"processed_{file_name}")
    processed_data.to_csv(processed_file_location, index=False)
    
    return processed_file_location


@app.post("/api/v1/predict")
async def upload_csv(file: UploadFile = File(...)):
    """
    업로드된 CSV 파일을 처리하는 FastAPI 엔드포인트
    """
    processed_file_location = await process_csv(file)
    
    return {"filename": file.filename, "message": "File successfully uploaded and processed!", "processed_file": processed_file_location}
    
    """
    AI 예측 수행 및 결과 반환
    """