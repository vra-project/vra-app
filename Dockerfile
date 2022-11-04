# syntax=docker/dockerfile:1
FROM python:3.9.13

WORKDIR /app
COPY requirements.txt requirements.txt
COPY ".streamlit" ".streamlit"
COPY pages pages

RUN pip3 install -r requirements.txt

COPY Game_Based.py Game_Based.py

# Expose port 
ENV PORT 8501

CMD streamlit run Game_Based.py