from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import preprocessing  # preprocessing.py 모듈을 임포트

app = Flask(__name__)

# 파일 저장 위치 설정
UPLOAD_DIR = "uploads/uploaded_files"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

PROCESSED_DIR = "uploads/processed_files"
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

def process_csv(file):
    """
    파일을 저장하고 전처리 후 처리된 파일을 반환하는 메서드
    """
    # 파일 이름 추출
    file_name = file.filename
    file_location = os.path.join(UPLOAD_DIR, file_name)
    
    # 파일을 메모리에서 읽고 Pandas로 처리하기
    file.save(file_location)  # Flask는 파일을 직접 디스크에 저장합니다.
    
    # pandas로 CSV 읽기
    data = pd.read_csv(file_location)
    
    # preprocessing.py의 main() 함수 호출하여 처리된 데이터 반환
    processed_data = preprocessing.main(file_location)  # 파일 경로를 전달
    
    # 저장된 처리된 파일 경로
    processed_file_location = os.path.join(PROCESSED_DIR, f"processed_{file_name}")
    processed_data.to_csv(processed_file_location, index=False)
    
    return processed_file_location

@app.route('/')
def index():
    """
    업로드 폼을 표시하는 홈 화면 엔드포인트
    """
    return render_template('index.html')

@app.route("/api/v1/predict", methods=["POST"])
def upload_csv():
    """
    업로드된 CSV 파일을 처리하는 Flask 엔드포인트
    """
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    
    processed_file_location = process_csv(file)
    
    return jsonify({
        "filename": file.filename,
        "message": "File successfully uploaded and processed!",
        "processed_file": processed_file_location
    })

if __name__ == "__main__":
    app.run(debug=True)
