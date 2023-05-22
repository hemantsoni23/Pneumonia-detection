FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
COPY app.py .
COPY model.h5 .
COPY animation2.json .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
# docker build -t pneumonia-detection .
# docker run -p 8501:8501 pneumonia-detection
