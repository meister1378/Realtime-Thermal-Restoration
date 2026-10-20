FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/ model/
COPY train/ train/
COPY infer/ infer/
COPY prompts/ prompts/
COPY checkpoints/ checkpoints/

WORKDIR /app/infer
ENTRYPOINT ["python", "infer.py"]
