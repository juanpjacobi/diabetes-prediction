from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_model(X_train, y_train, preprocessor):
    """Entrena modelo con GridSearchCV."""
    pipeline = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_.named_steps['classifier']
    print(f"🌟 Mejor combinación: {grid.best_params_}")
    return grid, best_model


def adjust_threshold(best_model, full_pipeline, X_test, y_test, threshold=0.20):
    """Ajusta umbral de clasificación para clases 1 y 2."""
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    # Usamos el preprocessor ya entrenado que está dentro del pipeline
    X_test_scaled = full_pipeline.named_steps['preprocessing'].transform(X_test)
    y_proba = best_model.predict_proba(X_test_scaled)

    y_pred_custom = []

    for probs in y_proba:
        if probs[1] >= threshold or probs[2] >= threshold:
            y_pred_custom.append(1 if probs[1] > probs[2] else 2)
        else:
            y_pred_custom.append(0)

    print("\n📄 Reporte de Clasificación (umbral ajustado):\n")
    print(classification_report(y_test, y_pred_custom))
    
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_custom)
    plt.title(f"Matriz de Confusión – Umbral Ajustado ({threshold})")
    plt.tight_layout()
    plt.show()

