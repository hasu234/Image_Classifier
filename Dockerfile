FROM python:3.9

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

# Define the entry point
ENTRYPOINT ["python"]

# Set the default command to an empty string
CMD []
