from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def preprocess(df):
    """Prepara datos: divide X/y, escalado, split."""
    df = df.drop_duplicates()
    
    X = df.drop(columns='Class')
    y = df['Class']

    preprocessor = ColumnTransformer([
        ("scaler", StandardScaler(), X.columns)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test, preprocessor
