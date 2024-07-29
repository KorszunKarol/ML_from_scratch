from data_pipeline import DataPipeline
from svm import SVM
import numpy as np
import json


def find_best_params_polynomial(
    learning_rate,
    lambda_param,
    gamma_values,
    r,
    d_values,
    X_train,
    y_train,
    X_test,
    y_test,
):
    params_dict = {}
    best_accuracy = 0
    best_params = None

    for gamma in gamma_values:
        for d in d_values:
            print(
                f"Training with learning rate: {learning_rate}, lambda: {lambda_param}, gamma: {gamma}, r: {r}, and d: {d}"
            )
            svm = SVM(
                learning_rate=learning_rate,
                lambda_param=lambda_param,
                iters=500,
                kernel=SVM.polynomial_kernel,
                kernel_params={"gamma": gamma, "r": r, "d": d},
            )

            w, b = svm.fit(X_train, y_train, save_weights=False)
            np.save(
                f"weights/weights_{svm.learning_rate}_{svm.lambda_param}_{gamma}_{r}_{d}.npy",
                w,
            )
            np.save(
                f"weights/bias_{svm.learning_rate}_{svm.lambda_param}_{gamma}_{r}_{d}.npy",
                b,
            )

            predictions = svm.predict(X_test)
            accuracy = np.sum(predictions == y_test) / len(y_test)
            print(accuracy)
            params_dict[(gamma, d)] = accuracy

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (gamma, d)

    return best_params, best_accuracy, params_dict


def find_best_params_linear(
    learning_rate, lambda_param, C_values, X_train, y_train, X_test, y_test
):
    params_dict = {}
    best_accuracy = 0
    best_C = None

    for c in C_values:
        print(
            f"Training with learning rate: {learning_rate}, lambda: {lambda_param}, and C: {c}"
        )
        svm = SVM(
            learning_rate=learning_rate,
            lambda_param=lambda_param,
            iters=500,
            kernel=SVM.linear_kernel,
            kernel_params={"c": c},
        )

        w, b = svm.fit(X_train, y_train, save_weights=False)
        np.save(f"weights/weights_{svm.learning_rate}_{svm.lambda_param}_{c}.npy", w)
        np.save(f"weights/bias_{svm.learning_rate}_{svm.lambda_param}_{c}.npy", b)

        predictions = svm.predict(X_test)
        accuracy = np.sum(predictions == y_test) / len(y_test)
        print(accuracy)
        params_dict[c] = accuracy

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_C = c

    return best_C, best_accuracy, params_dict


def find_best_params(learning_rates, lambda_params, X_train, y_train, X_test, y_test):
    params_dict = {}
    best_accuracy = 0
    best_params = None

    for lr in learning_rates:
        for lp in lambda_params:
            print(f"Training with learning rate: {lr} and lambda: {lp}")
            svm = SVM(learning_rate=lr, lambda_param=lp, iters=500)
            svm.fit(X_train, y_train, save_weights=True)
            predictions = svm.predict(X_test)
            accuracy = np.sum(predictions == y_test) / len(y_test)
            print(accuracy)
            params_dict[(lr, lp)] = accuracy

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (lr, lp)

    return best_params, best_accuracy, params_dict


def main():

    pipeline = DataPipeline(
        "wine+quality/winequality-red.csv",
        "wine+quality/winequality-white.csv",
        test_size=0.2,
    )
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    lambda_params = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    X_train, X_test, y_train, y_test = pipeline.split_data()

    best_param, best_accuracy, params_dict = find_best_params_polynomial(
        learning_rate=0.0005,
        lambda_param=0.0005,
        gamma_values=[0.1, 0.5, 1, 5],
        r=1,
        d_values=[2, 3, 4, 5],
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )
    params_dict = {str(k): v for k, v in params_dict.items()}

    with open("params_sigmoid.json", "w") as f:
        json.dump(params_dict, f)

    print(f"Best params: {best_param}")
    print(f"Best accuracy: {best_accuracy}")
    print(params_dict)
    # svm.fit(X_train, y_train, save_weights=True)
    # w = np.load(f"weights_{svm.learning_rate}_{svm.lambda_param}.npy")
    # b = np.load(f"bias_{svm.learning_rate}_{svm.lambda_param}.npy")
    # svm.w = w
    # svm.b = b
    # predictions = svm.predict(X_test)
    # accuracy = np.sum(predictions == y_test) / len(y_test)
    # print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    main()
