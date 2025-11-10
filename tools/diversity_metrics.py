import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the previous directory of the current script directory
project_dir = os.path.dirname(current_dir)
# Add the project directory to the module search path
sys.path.append(project_dir)

from prepare.pretrain_base_models_indiv.pretrain_cnn_indiv import load_and_encode_fasta_data as cnn_load_and_encode_fasta_data

from tensorflow.keras.optimizers import Adam


class DiversityMetricsCalculator:
    def __init__(self):
        self.k_values = []
        self.correlation_coefficients = []
        self.mean_errors = []
        self.disagreement_measures = []
        self.q_statistics = []

    def calculate_diversity_metrics(self, dataset_path):
        """
        Independently calculate the diversity index between the predicted results of different pretrained models on a given dataset.

        Paras:
        dataset_path (str): path to the datasets。

        Return:
        dict: A dictionary containing different metric names and their corresponding pairwise relationship metric values for multiple models.
        """
        base_dir = '/EnDeep4mC/pretrained_models/indiv'
        cnn_model_path = os.path.join(base_dir, f'cnn_{os.path.basename(dataset_path)}.h5')
        blstm_model_path = os.path.join(base_dir, f'blstm_{os.path.basename(dataset_path)}.h5')
        transformer_model_path = os.path.join(base_dir, f'transformer_{os.path.basename(dataset_path)}.h5')

        # Load pre trained model
        cnn_model = load_model(cnn_model_path)
        blstm_model = load_model(blstm_model_path)
        transformer_model = load_model(transformer_model_path)

        # Compile model (compilation parameters can be adjusted as needed)
        cnn_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        blstm_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
        transformer_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

        # Get the test set data
        _, _, X_test, _, y_test = cnn_load_and_encode_fasta_data(dataset_path)
        X_test = X_test.astype(np.float32)
        X_test = np.expand_dims(X_test, axis=1)
        X_test = np.repeat(X_test, 3, axis=1)

        # Obtain the prediction results of each model for the test set and convert them into category representations with a threshold of 0.5
        cnn_test_pred = (cnn_model.predict(X_test).reshape(-1, 1) > 0.5).astype(int)
        blstm_test_pred = (blstm_model.predict(X_test).reshape(-1, 1) > 0.5).astype(int)
        transformer_test_pred = (transformer_model.predict(X_test).reshape(-1, 1) > 0.5).astype(int)

        k_values = []
        correlation_coefficients = []
        mean_errors = []
        disagreement_measures = []
        q_statistics = []

        colors = ['r', 'g', 'c']  # Corresponding to the colors of three classifier combinations
        labels = ['CNN - BLSTM', 'CNN - Transformer', 'BLSTM - Transformer']  # Legend labels corresponding to colors

        plt.clf()

        for idx, pair in enumerate([(cnn_test_pred, blstm_test_pred), (cnn_test_pred, transformer_test_pred), (blstm_test_pred, transformer_test_pred)]):
            pred_1, pred_2 = pair
            a = 0
            b = 0
            c = 0
            d = 0

            for j in range(len(y_test)):
                # Count the corresponding quantity based on the combination of the predicted results from two classifiers
                if pred_1[j] == 1:
                    if pred_2[j] == 1:
                        if y_test[j] == 1:
                            a += 1
                        else:
                            b += 1
                    else:
                        if y_test[j] == 1:
                            c += 1
                        else:
                            d += 1
                else:
                    if pred_2[j] == 1:
                        if y_test[j] == 1:
                            c += 1
                        else:
                            d += 1
                    else:
                        if y_test[j] == 1:
                            a += 1
                        else:
                            b += 1

            # Calculate the relevant statistics for each pair of classifiers
            k_value = self.calculate_k_statistic(a, b, c, d)
            correlation_coefficient = self.calculate_correlation_coefficient(a, b, c, d)
            mean_error = self.calculate_mean_error(pred_1, pred_2, y_test)
            disagreement_measure = (b + c) / len(y_test) if len(y_test) > 0 else 0
            q_statistic = self.calculate_q_statistic(a, b, c, d)

            k_values.append(k_value)
            correlation_coefficients.append(correlation_coefficient)
            mean_errors.append(mean_error)
            disagreement_measures.append(disagreement_measure)
            q_statistics.append(q_statistic)

            # Draw the points corresponding to each pair of classifiers onto a scatter plot
            plt.scatter(k_value, mean_error, c=colors[idx], label=labels[idx])

        self.k_values = k_values
        self.correlation_coefficients = correlation_coefficients
        self.mean_errors = mean_errors
        self.disagreement_measures = disagreement_measures
        self.q_statistics = q_statistics

        return {
            "K Values": [(k_values[0], k_values[1], k_values[2], k_values[3], k_values[4], k_values[5])],
            "Correlation Coefficients": [(correlation_coefficients[0], correlation_coefficients[1],
                                          correlation_coefficients[2], correlation_coefficients[3],
                                          correlation_coefficients[4], correlation_coefficients[5])],
            "Mean Errors": [(mean_errors[0], mean_errors[1], mean_errors[2], mean_errors[3], mean_errors[4],
                             mean_errors[5])],
            "Disagreement Measures": [(disagreement_measures[0], disagreement_measures[1],
                                       disagreement_measures[2], disagreement_measures[3],
                                       disagreement_measures[4], disagreement_measures[5])],
            "Q - Statistics": [(q_statistics[0], q_statistics[1], q_statistics[2], q_statistics[3],
                               q_statistics[4], q_statistics[5])]
        }

    def calculate_k_statistic(self, a, b, c, d):
        """
        Calculate the K statistic index

        Paras:
        a, b, c, d (int): The statistical quantity corresponding to the combination of classifier prediction results。

        Return:
        float: the value of K。
        """
        total = a + b + c + d
        if total == 0:
            return 0
        p_1 = (a + c) / total
        p_2 = (a + b) / total
        if (1 - p_2) == 0:
            return 0
        return (p_1 - p_2) / (1 - p_2)

    def calculate_correlation_coefficient(self, a, b, c, d):
        """
        Calculate correlation coefficient

        Paras:
        a, b, c, d (int): The statistical quantity corresponding to the combination of classifier prediction results.

        返回:
        float: the value of correlation coefficient
        """
        numerator = a * d - b * c
        denominator = np.sqrt((a + b) * (a + c) * (c + d) * (b + d))
        if denominator == 0:
            return 0
        return numerator / denominator

    def calculate_mean_error(self, pred_1, pred_2, y_true):
        """
        Calculate the mean error

        Paras:
        pred_1 (numpy.ndarray): The prediction result of the first classifier (converted into category representation), with a shape of (sample size, 1)
        pred_2 (numpy.ndarray): The prediction result of the second classifier (converted to category representation) has a shape of (sample size, 1).
        y_true (numpy.ndarray): The real label has a shape of (sample size,).

        Return:
        float: value of the mean error
        """
        errors = []
        for i in range(len(y_true)):
            error_1 = np.abs(pred_1[i] - y_true[i])
            error_2 = np.abs(pred_2[i] - y_true[i])
            errors.append((error_1 + error_2) / 2)
        return np.mean(errors) if errors else 0

    def calculate_q_statistic(self, a, b, c, d):
        """
        Calculate Q the statistic value.

        Paras:
        a, b, c, d (int): The statistical quantity corresponding to the combination of classifier prediction results.

        Return:
        float: value of Q.
        """
        numerator = a * d - b * c
        denominator = a * d + b * c
        if denominator == 0:
            return 0
        return numerator / denominator

    def plot_and_save_k_values(self, dataset_path, save_dir='/EnDeep4mC/evaluations/visualization'):
        """
        Draw a scatter plot of K value and average error and save it. The file name is generated based on the dataset name to avoid overwriting.

        Paras:
        dataset_path (str): The path of the dataset, used to generate file names.
        save_dir (str): The default directory for saving images is the specified directory.
        """
        dataset_name = os.path.basename(dataset_path)
        plt.xlabel('K Value')
        plt.ylabel('Mean Error')
        plt.title(f'K - Mean Error_{dataset_name}')

        plt.legend()

        save_path = os.path.join(save_dir, f'k_mean_error_plot_{dataset_name}.png')
        plt.savefig(save_path)


if __name__ == "__main__":
    dataset_paths = ['4mC_A.thaliana', '4mC_C.elegans', '4mC_D.melanogaster',
                     '4mC_E.coli', '4mC_G.subterraneus', '4mC_G.pickeringii']
    #dataset_paths = ['4mC_E.coli']
    calculator = DiversityMetricsCalculator()
    all_metrics_data = []  # collect metric data for all datasets

    for dataset_path in dataset_paths:
        print(f"Calculating diversity measure value for dataset {dataset_path}")
        metrics_dict = calculator.calculate_diversity_metrics(dataset_path)

        print(f"Dataset: {dataset_path}")
        print("Measure values            CNN - BLSTM      CNN - Transformer     BLSTM - Transformer")
        print("-" * 150)
        metric_data = []  # collect metric data for the current dataset
        for metric_name, metric_values in metrics_dict.items():
            metric_row = [metric_name] + [format(value, '.6f') for value in metric_values[0]]
            print(f"{metric_row[0].ljust(25)} {metric_row[1].ljust(20)} {metric_row[2].ljust(20)} {metric_row[3].ljust(20)} "
                  f"{metric_row[4].ljust(20)} {metric_row[5].ljust(20)} {metric_row[6].ljust(20)}")
            metric_data.append(metric_row)
        all_metrics_data.append(metric_data)

        print("Drawing K-Mean Error Figure...")
        calculator.plot_and_save_k_values(dataset_path)
        print("The figure has been saved.")
