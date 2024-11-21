# 1. 베이스 이미지 설정: Python 3.9 버전 사용
FROM python:3.9-slim

# 2. 작업 디렉토리 생성
WORKDIR /app

# 3. 필수 라이브러리 설치
# 시스템 패키지 설치 (필요한 경우)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. 필요한 Python 라이브러리들 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 애플리케이션 코드 복사
COPY . .

# 6. FastAPI와 Uvicorn 설치
RUN pip install fastapi uvicorn

# 7. 실행: uvicorn으로 FastAPI 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
