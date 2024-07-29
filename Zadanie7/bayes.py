import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from enum import Enum
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class ScalerType(Enum):
    STANDARD = StandardScaler
    MINMAX = MinMaxScaler
    ROBUST = RobustScaler


def train_and_evaluate_models(X_train, X_test, y_train, y_test):

    models = {
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
    }
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = calculate_accuracy(y_test, y_pred)
        models[model_name] = accuracy
        print(f"Accuracy with {model_name}: {accuracy:.4f}")

    return models


def train(X_train, y_train):
    classes = np.unique(y_train)
    means = {c: X_train[y_train == c].mean(axis=0) for c in classes}
    stds = {c: X_train[y_train == c].std(axis=0) for c in classes}
    priors = {c: (y_train == c).mean() for c in classes}
    return means, stds, priors


def gaussian_likelihood(x, mean, std):
    eps = 1e-6
    std = np.where(std == 0, eps, std)
    exponent = -((x - mean) ** 2) / (2 * (std**2))
    return np.exp(exponent) / (np.sqrt(2 * np.pi) * std)


def log_gaussian_likelihood(x, mean, std):
    eps = 1e-6
    std = np.where(std == 0, eps, std)
    log_exponent = -((x - mean) ** 2) / (2 * (std**2))
    return log_exponent - np.log(np.sqrt(2 * np.pi) * std)


def predict(X, means, stds, priors):
    y_pred = []
    classes = list(means.keys())
    for x in X:
        log_posteriors = []
        for c in classes:
            log_likelihood = np.sum(log_gaussian_likelihood(x, means[c], stds[c]))
            log_prior = np.log(priors[c])
            log_posterior = log_likelihood + log_prior
            log_posteriors.append(log_posterior)
        y_pred.append(classes[np.argmax(log_posteriors)])
    return np.array(y_pred)


def calculate_accuracy(y_pred, y_test):
    return (y_pred == y_test).mean()


def scale_input_data(X, scaler_type=ScalerType):
    scaler = scaler_type.value()
    return scaler.fit_transform(X)


def main():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_scaled = scale_input_data(X, ScalerType.STANDARD)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=4, shuffle=True
    )
    means, stds, priors = train(X_train, y_train)
    y_pred = predict(X_test, means, stds, priors)
    accuracy = calculate_accuracy(y_pred, y_test)
    accuracy_2 = calculate_accuracy(predict(X_train, means, stds, priors), y_train)

    print("Accuracy train", accuracy_2)
    print("Accuracy:", accuracy)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", annot_kws={"size": 16})
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("True", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig("confusion_matrix.png", dpi=600)
    plt.show()

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    models = train_and_evaluate_models(X_train, X_test, y_train, y_test)
    print(models)


if __name__ == "__main__":
    main()
