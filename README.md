# Decision Trees in CV Experiments

Репозиторий посвящен исследованию способов интеграции решающих деревьев в архитектуры глубокого обучения для задач компьютерного зрения.

**Стек технологий:** Python, PyTorch, Hydra, MLflow, Optuna, uv.

## Запуск MLflow (Дашборд)

Для просмотра результатов экспериментов, графиков обучения и метрик (accuracy, FPS, VRAM, MACs) запустите сервер MLflow в отдельном окне терминала:

```bash
uv run mlflow ui --port 5000
```

> Дашборд будет доступен в браузере по адресу: http://127.0.0.1:5000

## Запуск экспериментов

### 1. Подбор гиперпараметров для NODE (Optuna)

Поиск оптимальных параметров:

- layer_dim (int):
  - Количество деревьев в одном слое.
  - При передаче в ODST этот параметр маппится на аргумент num_trees.
- num_layers (int):
  - Количество слоев из деревьев (блоков ODST), которые будут идти друг за другом в DenseBlock
- depth (int):
  - Глубина каждого дерева.
  - Количество листьев в таком дереве равно $2^{depth}$

```bash
uv run train.py -m model=simple_conv_node_se \
+hparams_search=node_optuna \
experiment_name="hparams_simple_conv_node"
```

### 2. Сравнение архитектур: Squeeze-and-Excitation vs NODE

Сравнительный запуск Baseline, Classic SE и Node SE в одинаковых условиях.

**Для датасета CIFAR-10:**

```bash
uv run train.py -m \
  model=simple_conv_baseline,simple_conv_classic_se,simple_conv_node_se \
  dataset=cifar10 \
  experiment_name=simple_conv_cifar10
```

**Для датасета CIFAR-100:**

```bash
uv run train.py -m \
  model=simple_conv_baseline,simple_conv_classic_se,simple_conv_node_se \
  dataset=cifar100 \
  experiment_name=simple_conv_cifar100
```

## Полезные флаги

Если нужно быстро проверить работоспособность кода без долгого обучения, можно ограничить размер датасета:

```bash
# Добавьте эти флаги в конец любой команды запуска:
dataset.train_size=500 dataset.test_size=50
```
