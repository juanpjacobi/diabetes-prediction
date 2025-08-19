import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay

def evaluate_model(model, grid, X_test, y_test):
    """Evalúa el modelo base y muestra métricas."""
    print("\n📄 Reporte de Clasificación:")
    y_pred = grid.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Precisión del modelo: {accuracy_score(y_test, y_pred):.4f}")

    ConfusionMatrixDisplay.from_estimator(grid, X_test, y_test)
    plt.title("Matriz de Confusión – Random Forest")
    plt.tight_layout()
    plt.show()

    importances = model.feature_importances_
    features = X_test.columns
    features_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by="Importance", ascending=False)

    print("\n🔥 Top 10 características más importantes:")
    print(features_df.head(10))

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=features_df.head(15), palette='mako')
    plt.title('Importancia de características por Random Forest')
    plt.tight_layout()
    plt.show()
