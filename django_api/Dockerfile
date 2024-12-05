# Backend Dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

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

# Copy project files
COPY . /api

# Collect static files
# RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Run the application
# CMD ["gunicorn", "--bind", "0.0.0.0:8000", "your_project_name.wsgi:application"]
# ENTRYPOINT ["python"]
CMD ["python" "manage.py" "runserver"]