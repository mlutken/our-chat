import torch
from torch.cuda.amp import autocast, GradScaler

# Check GPU and PyTorch compatibility
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"SM version: {torch.cuda.get_device_capability(0)}")  # Should be (8, 9) for RTX 4060

# Initialize model and optimizer
model = torch.nn.Linear(10, 10).cuda()
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler(enabled=True)  # For FP16

# Training step with AMP
inputs = torch.randn(32, 10).cuda()
targets = torch.randn(32, 10).cuda()

with autocast(enabled=True, dtype=torch.float16):  # or torch.bfloat16
    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()
