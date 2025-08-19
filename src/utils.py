import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """Carga el dataset desde un archivo CSV."""
    return pd.read_csv(path)
