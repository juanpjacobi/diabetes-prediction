import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    """Ejecuta EDA básico con gráficos de distribución y correlación."""
    print(df.info())
    print(df.describe())
    print("\nDistribución de clases:")
    print(df['Class'].value_counts(normalize=True))
    
    sns.countplot(data=df, x="Class", palette="Set2")
    plt.title("Distribución de clases")
    plt.grid(axis="y")
    plt.show()

    df.drop(columns="Class").hist(bins=20, figsize=(16, 10), color="skyblue", edgecolor="black")
    plt.suptitle("Distribuciones de variables numéricas", fontsize=16)
    plt.tight_layout()
    plt.show()

    features = df.columns.drop("Class")
    plt.figure(figsize=(20, 20))
    for i, col in enumerate(features, 1):
        plt.subplot(4, 3, i)
        sns.boxplot(data=df, x="Class", y=col, palette="pastel")
        plt.title(f"{col} por clase")
        plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matriz de correlación")
    plt.show()

    correlaciones = df.corr(numeric_only=True)["Class"].sort_values(ascending=False)
    print("📌 Correlación de cada variable con la clase:\n")
    print(correlaciones)
