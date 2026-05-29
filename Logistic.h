#ifndef LOGISTIC_REGRESSION_HPP
#define LOGISTIC_REGRESSION_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "base_model.h"
#include "loss_model.h"

namespace sklearn_cpp {
    namespace linear_model {

        class LogRegBinary : public BaseModel {

            protected:
                LossModel& loss_model;

                double sigmoid(const double& z) const {
                    return 1.0 / (1.0 + std::exp(-z));
                }

                double Accuracy(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) const {
                    size_t correct{0};
                    for (size_t i = 0; i < Y.size(); i++) {
                        double z{b};
                        for (size_t j = 0; j < w.size(); j++) {
                            z += w[j] * X[i][j];
                        }
                        double prediction = (sigmoid(z) >= 0.5) ? 1.0 : 0.0;
                        if (prediction == Y[i]) correct++;
                    }
                    return static_cast<double>(correct) / Y.size();
                }

                void compute_gradients(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, std::vector<double>& grad_w, double& grad_b) override {
                    size_t m{Y.size()};
                    for (size_t i = 0; i < m; i++) {
                        double z{b};
                        for (size_t j = 0; j < w.size(); j++) {
                            z += w[j] * X[i][j];
                        }
                        double error = sigmoid(z) - Y[i];
                        for (size_t j = 0; j < w.size(); j++) {
                            grad_w[j] += error * X[i][j];
                        }
                        grad_b += error;
                    }
                    double scale = 1.0 / static_cast<double>(m);
                    for (size_t j = 0; j < w.size(); j++) {
                        grad_w[j] *= scale;
                    }
                    grad_b *= scale;
                }

            public:
                LogRegBinary(int n_features, LossModel& loss, double lr = 0.01, int iter = 1000)
                    : BaseModel(n_features, lr, iter), loss_model(loss) {}

                std::vector<double> predict(const std::vector<std::vector<double>>& X) const override {
                    std::vector<double> predictions;
                    predictions.reserve(X.size());
                    for (size_t i = 0; i < X.size(); i++) {
                        double z{b};
                        for (size_t j = 0; j < w.size(); j++) {
                            z += w[j] * X[i][j];
                        }
                        predictions.push_back(sigmoid(z) >= 0.5 ? 1.0 : 0.0);
                    }
                    return predictions;
                }

                std::vector<double> predictProbability(const std::vector<std::vector<double>>& X) const {
                    std::vector<double> probabilities;
                    probabilities.reserve(X.size());
                    for (size_t i = 0; i < X.size(); i++) {
                        double z{b};
                        for (size_t j = 0; j < w.size(); j++) {
                            z += w[j] * X[i][j];
                        }
                        probabilities.push_back(sigmoid(z));
                    }
                    return probabilities;
                }

                void printLoss(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) const {
                    double loss = loss_model.computeLoss(X, Y, w, b);
                    double accuracy = Accuracy(X, Y);
                    std::cout << std::fixed << std::setprecision(4);
                    std::cout << "=====Logistic REGRESSION RESULTS=====\n";
                    std::cout << "Binary Loss: " << loss << "\n";
                    std::cout << "Accuracy: " << accuracy * 100.0 << "%\n";
                }

                void printPara() const {
                    std::cout << std::fixed << std::setprecision(6);
                    std::cout << "LEARNED PARAMETERS\n";
                    for (size_t j = 0; j < w.size(); j++) {
                        std::cout << "w[" << j << "] = " << w[j] << "\n";
                    }
                    std::cout << "b = " << b << "\n";
                }
        };

    }
}

#endif