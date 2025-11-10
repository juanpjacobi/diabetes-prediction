from utils import load_data
from eda import run_eda
from preprocessing import preprocess
from modeling import train_model, adjust_threshold
from evaluation import evaluate_model

def main():
    # 1. Cargar datos
    df = load_data("./data/diabetes.csv")

    # 2. EDA
    run_eda(df)

    # 3. Preprocesamiento
    X_train, X_test, y_train, y_test, preprocessor = preprocess(df)

    # 4. Entrenamiento
    grid, best_model = train_model(X_train, y_train, preprocessor)

    # 5. Evaluación
    evaluate_model(best_model, grid, X_test, y_test)

    # 6. Ajuste de umbral
    adjust_threshold(best_model, grid.best_estimator_, X_test, y_test, threshold=0.20)


if __name__ == "__main__":
    main()
