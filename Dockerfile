FROM python:3.8

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

# Define the entry point
ENTRYPOINT ["python"]

# Set the default command to an empty string
CMD []
