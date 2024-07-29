import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_pipeline import DataPipeline
import json
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from svm import SVM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


@dataclass
class ModelComparison:
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame
    svm: SVM
    logistic: LogisticRegression = field(
        default_factory=lambda: LogisticRegression(max_iter=1000)
    )
    metrics: dict = field(default_factory=dict)
    X_dev: pd.DataFrame = field(default=None)
    y_dev: pd.DataFrame = field(default=None)

    def read_metrics_from_file(self):
        with open("metrics_fixed.json", "r") as f:
            self.metrics = json.load(f)

    def plot_confusion_matrix(self, model_type: str, save_to_file: bool = False):
        matrix = pd.DataFrame(
            confusion_matrix(
                self.y_test, self.metrics[model_type]["test"]["predictions"]
            )
        )
        sns.heatmap(matrix, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        if save_to_file:
            plt.savefig(f"confusion_matrix_{model_type}.png", dpi=600)
        plt.show()

    def fit_and_evaluate(self, save_to_file=False):

        self.logistic.fit(self.X_train, self.y_train)
        logistic_train_predictions = self.logistic.predict(self.X_train)
        logistic_test_predictions = self.logistic.predict(self.X_test)

        self.svm.fit(self.X_train, self.y_train)
        svm_train_predictions = self.svm.predict(self.X_train)
        svm_test_predictions = self.svm.predict(self.X_test)

        self.metrics = {
            "logistic": {
                "train": {
                    "accuracy": accuracy_score(
                        self.y_train, logistic_train_predictions
                    ),
                    "precision": precision_score(
                        self.y_train, logistic_train_predictions
                    ),
                    "recall": recall_score(self.y_train, logistic_train_predictions),
                    "f1": f1_score(self.y_train, logistic_train_predictions),
                    "predictions": logistic_train_predictions.tolist(),
                },
                "test": {
                    "accuracy": accuracy_score(self.y_test, logistic_test_predictions),
                    "precision": precision_score(
                        self.y_test, logistic_test_predictions
                    ),
                    "recall": recall_score(self.y_test, logistic_test_predictions),
                    "f1": f1_score(self.y_test, logistic_test_predictions),
                    "predictions": logistic_test_predictions.tolist(),
                },
            },
            "svm": {
                "train": {
                    "accuracy": accuracy_score(self.y_train, svm_train_predictions),
                    "precision": precision_score(self.y_train, svm_train_predictions),
                    "recall": recall_score(self.y_train, svm_train_predictions),
                    "f1": f1_score(self.y_train, svm_train_predictions),
                    "predictions": svm_train_predictions.tolist(),
                },
                "test": {
                    "accuracy": accuracy_score(self.y_test, svm_test_predictions),
                    "precision": precision_score(self.y_test, svm_test_predictions),
                    "recall": recall_score(self.y_test, svm_test_predictions),
                    "f1": f1_score(self.y_test, svm_test_predictions),
                    "predictions": svm_test_predictions.tolist(),
                },
            },
        }
        if save_to_file:
            with open("metrics_fixed.json", "w") as f:
                json.dump(self.metrics, f)
        return self.metrics


def main():
    pipeline = DataPipeline(
        "wine+quality/winequality-red.csv",
        "wine+quality/winequality-white.csv",
        test_size=0.2,
        dev_size=0,
    )
    data = pipeline.split_data()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    comparison = ModelComparison(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        svm=SVM(
            learning_rate=0.0005,
            lambda_param=0.0005,
            iters=1000,
            kernel=SVM.polynomial_kernel,
            kernel_params={"d": 3, "r": 0, "gamma": 1},
        ),
        logistic=LogisticRegression(max_iter=1000),
    )
    # comparison.fit_and_evaluate(save_to_file=True)
    comparison.read_metrics_from_file()
    comparison.plot_confusion_matrix("logistic", save_to_file=True)


if __name__ == "__main__":
    main()
