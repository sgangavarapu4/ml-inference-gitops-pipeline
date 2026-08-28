import torch
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="GPU Inference API")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

@app.get("/infer")
def infer():
    if not torch.cuda.is_available():
        return {"status": "error", "message": "CUDA non-functional"}
    
    x = torch.rand(1000, 1000, device="cuda")
    y = torch.matmul(x, x)
    return {
        "status": "success", 
        "tensor_sum": float(y.sum()),
        "device": torch.cuda.get_device_name(0)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
2. Create Dockerfile:

Bash
cat << 'EOF' > Dockerfile
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
EOF