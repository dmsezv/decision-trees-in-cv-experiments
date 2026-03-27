import hydra
from omegaconf import DictConfig, OmegaConf
import mlflow
import torch


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(cfg.experiment_name)

    device = torch.device(
        'mps' if torch.backends.mps.is_available() else \
        'cuda' if torch.cuda.is_available() else \
        'cpu'
    )
    print(f"Используемое устройство: {device}")

    
    with mlflow.start_run(run_name=cfg.model._target_.split('.')[-1]):
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        print(f"=== Запуск эксперимента: {cfg.experiment_name} ===")
        print(f"Конфигурация:\n{OmegaConf.to_yaml(cfg)}")

        pass

if __name__ == "__main__":
    main()