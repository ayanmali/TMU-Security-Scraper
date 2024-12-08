# Backend Dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV OPENAI_API_KEY OPENAI_API_KEY

# Set work directory
WORKDIR /api

# Install system dependencies
# RUN apt-get update && apt-get install -y \
#     postgresql-client \
#     build-essential \
#     libpq-dev

# Copy requirements file
COPY requirements.txt /api

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118

# Copy project files
COPY . .

# Collect static files
# RUN python manage.py collectstatic --noinput

# Expose port
# EXPOSE 8000

# Run the application
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "your_project_name.wsgi:application"]
# ENTRYPOINT ["python"]

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]