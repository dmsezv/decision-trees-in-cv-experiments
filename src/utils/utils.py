def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Списки просто превращаем в строку, чтобы они не ломали MLflow
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)