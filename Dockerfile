FROM python:3.10.14-bullseye

# Copy local code to the container image.
WORKDIR /app
COPY . ./

# Install dependencies.
RUN pip install -r requirements.txt

# Expose the port on which the application will run
EXPOSE 8000

# Run the FastAPI application using uvicorn server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--env-file", "custom_env_data.env"]