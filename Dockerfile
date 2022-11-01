# syntax=docker/dockerfile:1
FROM python:3.9.13

WORKDIR /app
COPY requirements.txt requirements.txt
COPY ".streamlit" ".streamlit"

RUN pip3 install -r requirements.txt

COPY vra_app.py vra_app.py

# Expose port 
ENV PORT 8501

CMD streamlit run vra_app.py