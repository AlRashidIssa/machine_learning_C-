#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cmath>
#include <ctime>
#include "Untitled-1.h"

using namespace std;

class DataProcessor
{
private:
    string filename;
    char delimiter;
    vector<vector<double>> data;
    vector<vector<double>> X_train, X_test;
    vector<double> y_train, y_test;

public:
    // Constructor
    DataProcessor(const string& filename, char delimiter = ',')
        : filename(filename), delimiter(delimiter) {}

    // Funcation to read data from CSV file
    void readData()
    {
        ifstream file(filename);
        if(!file.is_open())
        {
            cerr << "Error: Unable to open file " << filename << endl;
            return;
        }
        string line;
        while (getline(file, line))
        {
            istringstream iss(line);
            string token;
            vector<double> row;
            while (getline(iss, token, delimiter))
            {
                row.push_back(stod(token));
            }
            data.push_back(row);
        }
        file.close();
    }// Funcation to read data from CSV file.End
    // Function to preprocess data
    void preprocessData() {
        // Scaling
        for (size_t i = 0; i < data[0].size(); ++i) {
            double min_val = data[0][i];
            double max_val = data[0][i];
            for (size_t j = 0; j < data.size(); ++j) {
                min_val = min(min_val, data[j][i]);
                max_val = max(max_val, data[j][i]);
            }
            double range = max_val - min_val;
            if (range == 0) // Avoid division by zero
                range = 1.0;
            for (size_t j = 0; j < data.size(); ++j) {
                data[j][i] = (data[j][i] - min_val) / range;
            }
        }

        // Normalization
        for (size_t i = 0; i < data[0].size(); ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < data.size(); ++j) {
                sum += data[j][i];
            }
            double mean = sum / data.size();

            double variance = 0.0;
            for (size_t j = 0; j < data.size(); ++j) {
                variance += (data[j][i] - mean) * (data[j][i] - mean);
            }
            double std_dev = sqrt(variance / data.size());

            if (std_dev == 0) // Avoid division by zero
                std_dev = 1.0;
            for (size_t j = 0; j < data.size(); ++j) {
                data[j][i] = (data[j][i] - mean) / std_dev;
            }
        }
    }// Function to preprocess data.End

    // Function to split data into training and testing sets
    // Function to split data into training and testing sets
    void splitData(double test_size) {
        size_t test_data_size = static_cast<size_t>(test_size * data.size());
        std::shuffle(data.begin(), data.end(), std::mt19937(std::random_device()()));
        X_train.assign(data.begin(), data.end() - test_data_size);
        X_test.assign(data.end() - test_data_size, data.end());

        y_train.reserve(X_train.size());
        y_test.reserve(X_test.size());

        for (size_t i = 0; i < X_train.size(); ++i) {
            y_train.push_back(X_train[i].back());
            X_train[i].pop_back();
        }

        for (size_t i = 0; i < X_test.size(); ++i) {
            y_test.push_back(X_test[i].back());
            X_test[i].pop_back();
        }
    }// Function to split data into training and testing sets.End 
    // Getters for training and testing data
    vector<vector<double>>& getXTrain() { return X_train; }
    vector<vector<double>>& getXTest() { return X_test; }
    vector<double>& getYTrain() { return y_train; }
    vector<double>& getYTest() { return y_test; }

};//DataProcessor.End

class ClassificationMetrics 
{
private:
    vector<double> true_labels;
    vector<double> predicted_labels;

public:
    // Constructor
    ClassificationMetrics(const vector<double>& true_labels, const vector<double>& predicted_labels)
        : true_labels(true_labels), predicted_labels(predicted_labels) {}

    // Function to calculate accuracy
    double accuracy() {
        int correct_predictions = 0;
        for (size_t i = 0; i < true_labels.size(); ++i) {
            if (true_labels[i] == predicted_labels[i]) {
                correct_predictions++;
            }
        }

        return static_cast<double>(correct_predictions) / true_labels.size();
    }

    // Function to calculate precision
    double precision(double class_label) {
        int true_positives = count_if(true_labels.begin(), true_labels.end(), [&](double true_label) { return true_label == class_label; });
        int false_positives = count_if(predicted_labels.begin(), predicted_labels.end(), [&](double predicted_label) { return predicted_label == class_label; });
        return static_cast<double>(true_positives) / (true_positives + false_positives);
    }

    // Function to calculate recall
    double recall(double class_label) {
        int true_positives = count_if(true_labels.begin(), true_labels.end(), [&](double true_label) { return true_label == class_label; });
        int false_negatives = count_if(predicted_labels.begin(), predicted_labels.end(), [&](double predicted_label) { return predicted_label != class_label; });
        return static_cast<double>(true_positives) / (true_positives + false_negatives);
    }

    // Function to calculate F1-score
    double f1_score(double class_label) {
        double precision_value = precision(class_label);
        double recall_value = recall(class_label);
        if (precision_value == 0 || recall_value == 0) {
            return 0; // Avoid division by zero
        }
        return 2 * (precision_value * recall_value) / (precision_value + recall_value);
    }
}; // ClassificationMetrics.End

class LogisticRegression {
private:
    vector<double> weights;
    double bias;
    int num_features;

    // Initialize weights to small random values
    void initializeWeights() {
        weights.resize(num_features);
        srand(time(nullptr)); // Seed random number generator
        for (int i = 0; i < num_features; ++i) {
            weights[i] = 0.01 * (rand() % 1000 - 500);
        }
        bias = 0.0;
    }

    // Sigmoid activation function
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

public:
    // Constructor
    LogisticRegression(int num_features) : num_features(num_features) {
        // Initialize weights and bias
        initializeWeights();
    }

    // Predict the class label for a single instance
    double predict(const vector<double>& instance) {
        double score = bias;
        for (int i = 0; i < num_features; ++i) {
            score += weights[i] * instance[i];
        }
        return sigmoid(score);
    }

    // Train the model using gradient descent
    void train(const vector<vector<double>>& X_train, const vector<double>& y_train, double learning_rate, int number_iterations) {
        for (int iter = 0; iter < number_iterations; ++iter) {
            double bias_gradient = 0.0;
            vector<double> weights_gradient(num_features, 0.0);

            for (size_t i = 0; i < X_train.size(); ++i) {
                double prediction = predict(X_train[i]);
                double error = prediction - y_train[i];
                bias_gradient += error;
                for (int j = 0; j < num_features; ++j) {
                    weights_gradient[j] += error * X_train[i][j];
                }
            }

            // Update bias and weights
            bias -= learning_rate * bias_gradient / X_train.size();
            for (int j = 0; j < num_features; ++j) {
                weights[j] -= learning_rate * weights_gradient[j] / X_train.size();
            }
        }// loop iter.End
    }// Train Fun.End
}; // LogisticRegression.End

int main()
{
    // Example usage
    string filename = "train.csv";
    char delimiter = ',';
    double test_size = 0.2; // 20% of data will be used for testing
    double learning_rate = 0.01;
    int num_iterations = 1000;

    // Create DataProcessor object and read data from CSV file
    DataProcessor dataProcessor(filename, delimiter);
    dataProcessor.readData();
    dataProcessor.preprocessData();
    dataProcessor.splitData(test_size);
    vector<vector<double>>& X_train = dataProcessor.getXTrain();
    vector<vector<double>>& X_test = dataProcessor.getXTest();
    vector<double>& y_train = dataProcessor.getYTrain();
    vector<double>& y_test = dataProcessor.getYTest();

    // Create LogisticRegression object
    int num_features = X_train[0].size();
    LogisticRegression logisticRegression(num_features);

    // Train the logistic regression model
    logisticRegression.train(X_train, y_train, learning_rate, num_iterations);


    // Make predictions on the test set
    vector<double> predictions;
    for (const auto& instance : X_test) {
        predictions.push_back(logisticRegression.predict(instance));
    }

    // Evaluate model performance
    ClassificationMetrics metrics(y_test, predictions);
    cout << "Accuracy: " << metrics.accuracy() << endl;
    cout << "Precision (class 1): " << metrics.precision(1) << endl;
    cout << "Recall (class 1): " << metrics.recall(1) << endl;
    cout << "F1-score (class 1): " << metrics.f1_score(1) << endl;

    return 0;
}
    
 