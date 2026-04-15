FROM python:3.10

# Set up operational directory
WORKDIR /app

# Copy system dependency trackers
COPY requirements.txt .

# Optimize local caching and explicitly map architecture packages
RUN pip install --no-cache-dir -r requirements.txt

# Integrate Core Codebase natively
COPY . .

# Run native async web server logic mapped to explicit internal Render Port (10000)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
