from python:3.11
EXPOSE 8083
workdir /app
COPY requirements.txt ./
run pip install -r requirements.txt
copy . ./
ENTRYPOINT ["streamlit","run","admin.py","--server.port=8083","server.address=0.0.0.0"]
