import time

import torch
from thop import profile


def log_macs_and_params(model, device, img_size=(3, 32, 32)):
    """Считает и возвращает количество MACs и параметров модели."""
    dummy_input = torch.randn(1, *img_size).to(device)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    return macs, params


def measure_inference_fps(model, device, img_size=(3, 32, 32), batch_size=1, num_runs=100):
    model.eval()
    dummy_input = torch.randn(batch_size, *img_size).to(device)

    def synchronize():
        if str(device) == "mps":
            torch.mps.synchronize()
        elif str(device) == "cuda":
            torch.cuda.synchronize()

    # 1. Прогрев (Warm-up)
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    synchronize()

    # 2. Основной замер
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    fps = (num_runs * batch_size) / total_time

    return fps
