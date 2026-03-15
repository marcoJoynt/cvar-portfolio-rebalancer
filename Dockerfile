# CVaR Portfolio Rebalancer — Streamlit app for Railway
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY rebalancer/ rebalancer/
COPY app.py .

# Railway sets PORT; default for local runs
ENV PORT=8501
EXPOSE 8501

# Bind to 0.0.0.0 so Railway can reach the server
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true
