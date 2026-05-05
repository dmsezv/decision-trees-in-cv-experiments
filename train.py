import gc
import statistics
from pathlib import Path

import hydra
import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.training.trainer import train_validate_loop
from src.utils.metrics import log_macs_and_params, measure_inference_fps
from src.utils.utils import flatten_dict, set_seed


def _run_single_seed(cfg: DictConfig, seed: int, device: torch.device, parent_run_id: str) -> dict:
    set_seed(seed)
    train_loader, test_loader = hydra.utils.instantiate(cfg.dataset, seed=seed)

    sample_inputs, _ = next(iter(train_loader))
    img_size = tuple(sample_inputs.shape[1:])

    model = hydra.utils.instantiate(cfg.model).to(device)
    if hasattr(model, "initial_weights"):
        print("[INFO] Initial NODE weights")
        model.initial_weights(loader=train_loader, device=device)

    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=model.parameters())
    criterion = nn.CrossEntropyLoss()

    run_name = f"{cfg.model._target_.split('.')[-1]}__seed_{seed}"
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag("seed", seed)
        mlflow.set_tag("parent_run_id", parent_run_id)

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        flat_cfg = flatten_dict(cfg_dict)
        mlflow.log_params(flat_cfg)

        print(f"=== Start experiment: {cfg.experiment_name} | seed: {seed} ===")
        print(f"[INFO] Configuration:\n{OmegaConf.to_yaml(cfg)}")

        macs, params = log_macs_and_params(model, device, img_size)
        mlflow.log_params({"macs": macs, "trainable_params": params})

        metrics = train_validate_loop(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            opt=optimizer,
            crt=criterion,
            device=device,
            num_epochs=cfg.training.num_epochs,
            early_stopping_patience=cfg.training.early_stopping_patience,
        )

        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), checkpoint_dir / f"{run_name}.pt")

        fps_batch_1 = measure_inference_fps(model, device, batch_size=1)
        fps_batch_64 = measure_inference_fps(model, device, batch_size=64)

        mlflow.log_metrics({"fps_batch_1": fps_batch_1, "fps_batch_64": fps_batch_64})

        metrics["fps_batch_1"] = fps_batch_1
        metrics["fps_batch_64"] = fps_batch_64

    return metrics


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(cfg.experiment_name)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Device: {device}")

    run_name = cfg.model._target_.split(".")[-1]
    all_metrics = []

    with mlflow.start_run(run_name=f"{run_name}__multi_seed") as parent_run:
        mlflow.set_tag("multi_seed", True)
        mlflow.set_tag("seeds", str(list(cfg.seeds)))

        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        flat_cfg = flatten_dict(cfg_dict)
        mlflow.log_params(flat_cfg)

        for seed in cfg.seeds:
            metrics = _run_single_seed(cfg, seed, device, parent_run.info.run_id)
            all_metrics.append(metrics)
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        agg_keys = list(all_metrics[0].keys())
        agg_metrics = {}
        for k in agg_keys:
            values = [m[k] for m in all_metrics]
            agg_metrics[f"{k}_mean"] = statistics.mean(values)
            if len(values) > 1:
                agg_metrics[f"{k}_std"] = statistics.stdev(values)

        mlflow.log_metrics(agg_metrics)

    return agg_metrics["best_val_accuracy_mean"]


if __name__ == "__main__":
    main()
