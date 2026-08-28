
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

WORKDIR /app

ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/wsl/drivers
ENV TORCH_CUDA_ARCH_LIST="10.0+PTX"
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 && \
    pip install --no-cache-dir fastapi uvicorn

COPY apps/ml-inference/app.py /app/main.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
