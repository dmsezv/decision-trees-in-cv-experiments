import hydra
import mlflow
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from src.training.trainer import train_validate_loop


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(cfg.experiment_name)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"[INFO] Device: {device}")

    train_loader, test_loader = hydra.utils.instantiate(cfg.dataset)

    model = hydra.utils.instantiate(cfg.model).to(device)
    if hasattr(model, "initial_weights"):
        print("[INFO] Initial NODE weights")
        model.initial_weights(loader=train_loader, device=device)

    optimizer = hydra.utils.instantiate(cfg.training.optimizer, params=model.parameters())
    criterion = nn.CrossEntropyLoss()

    run_name = cfg.model._target_.split(".")[-1]
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        print(f"=== Start experiment: {cfg.experiment_name} ===")
        print(f"[INFO] Configuration:\n{OmegaConf.to_yaml(cfg)}")

        train_validate_loop(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            opt=optimizer,
            crt=criterion,
            device=device,
            num_epochs=cfg.training.num_epochs,
            early_stopping_patience=cfg.training.early_stopping_patience,
        )


if __name__ == "__main__":
    main()
