# 라즈베리파이(Debian bookworm, Python 3.11)와 같은 테스트 환경
FROM python:3.11-slim-bookworm

# libgl1, libglib2.0-0: OpenCV 창 표시용
# fonts-nanum: 한글 문구용
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["pytest"]
