from dataclasses import dataclass, field
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataPipeline:
    csv_file_path1: str
    csv_file_path2: str
    data: pd.DataFrame = field(init=False)
    test_size: float = 0.15
    dev_size: float = 0.15

    def __post_init__(self):
        data1 = pd.read_csv(self.csv_file_path1, sep=";")
        data1["label"] = 1.0
        data2 = pd.read_csv(self.csv_file_path2, sep=";")
        data2["label"] = -1.0
        self.data = pd.concat([data1, data2])

    def split_data(self, seed=42):
        X = self.data.drop("label", axis=1)
        y = self.data["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=seed
        )
        if self.dev_size > 0:
            X_train, X_dev, y_train, y_dev = train_test_split(
                X_train, y_train, test_size=self.dev_size, random_state=seed
            )
            return {
                "X_train": X_train,
                "X_dev": X_dev,
                "X_test": X_test,
                "y_train": y_train,
                "y_dev": y_dev,
                "y_test": y_test,
            }

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }


def main():

    pipeline = DataPipeline(
        "wine+quality/winequality-red.csv", "wine+quality/winequality-white.csv"
    )
    X_train, X_test, y_train, y_test = pipeline.split_data()
    print(X_train.dtypes)


if __name__ == "__main__":
    main()
