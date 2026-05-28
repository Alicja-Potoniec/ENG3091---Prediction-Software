#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include "Regression.h"



void Linear::compute_gradients(std::vector<double>& derive_w, double& derive_b) {
    for(size_t i = 0; i < x.size(); i++) {
        double y_calc {b};
        for (size_t j = 0; j < m; j++) {
            y_calc += w[j] * x[i][j];
        };

        double error = y_calc - y[i];

        for (size_t j = 0; j < m; j++) {
            derive_w[j] += error * x[i][j];
        };
        derive_b += error;
    }
    
    for (size_t j = 0; j < m; j++) {
        derive_w[j] /= x.size();
    };
    derive_b /= x.size();
};

void Logistic::compute_gradients(std::vector<double>& derive_w, double& derive_b) {
                for (size_t i = 0; i < x.size(); i++) {
                    double z{b};
                    
                    for (size_t j = 0; j < m; j++) {
                        z += w[j] * x[i][j];
                    }

                    double sigmoid = 1.0 / (1.0 + std::exp(-z) );

                    double error = sigmoid - y[i];

                    for (size_t j = 0; j < m; j++) {
                        derive_w[j] += error * x[i][j];
                    }
                    
                    derive_b += error;
                }
                
                for (size_t i = 0; i < m; i++) {
                    derive_w[i] /= x.size();
                }

                derive_b /= x.size();

            };

void MulticlassLogistic::compute_gradient(std::vector<std::vector<double>>& derive_w, 
            std::vector<double>& derive_b) {
                for (size_t i = 0; i < x.size(); i++) {
                    std::vector<double> z (K,0.0);

                    for (size_t j = 0; j < K; j++) {
                        z[j] = b[j];

                        for (size_t k = 0; k < m; k++) {
                            z[j] += w[j][k] * x[i][k];
                        };
                    };

                    std::vector<double> softmax(K, 0.0);
                    double denom {0.0};

                    for (size_t k = 0; k < K; k++) {
                        denom += std::exp(z[k]);
                    };
                    for (size_t k = 0; k < K; k++) {
                        softmax[k] = std::exp(z[k]) / denom;
                    };
                    
                    for (size_t k = 0; k < K; k++) {
                        double y_init {static_cast<size_t>(y[i]) == k ? 1.0 : 0.0};
                        double error {softmax[k] - y_init};
                        for (size_t j = 0; j < m; j++) {
                            derive_w[k][j] += error * x[i][j];
                        };

                        derive_b[k] += error;
                    };

                };

                for (size_t k = 0; k < K; k++) {
                    for (size_t j = 0; j < m; j++) {
                        derive_w[k][j] = derive_w[k][j] / x.size() + (2 * lambda * w[k][j]) ;
                    };
                    derive_b[k] = derive_b[k] / x.size() + (2 * lambda * b[k]);
                };
            };


std::vector<double> Linear::predict(const std::vector<std::vector<double>>& X) {
    std::vector<double> predictions;

    for (size_t i = 0; i < X.size(); i++) {
        double y {b};
        for (size_t j = 0; j < m; j++) {
            y += w[j] * X[i][j];
        };
        predictions.push_back(y);
    };
    return predictions;
};

std::vector<double> Logistic::predict(const std::vector<std::vector<double>>& X) {
    std::vector<double> predictions;

    for (size_t i = 0; i < X.size(); i++) {
        double z {b};
        for (size_t j = 0; j < m; j++) {
            z += w[j] * X[i][j];
        };
        double y {1.0 / (1.0 + std::exp(-z))};
        predictions.push_back(y > 0.5 ? 1.0 : 0.0);
    };
    return predictions;
}

std::vector<double> MulticlassLogistic::predict(const std::vector<std::vector<double>>& X) {
    std::vector<double> predictions;

    for (size_t i = 0; i < X.size(); i++) {
        std::vector<double> z (K,0.0);
        for (size_t j = 0; j < K; j++) {
            z[j] = b[j];
            for (size_t k = 0; k < m; k++) {
                z[j] += w[j][k] *  X[i][k];
            };
        };

        double sum{0.0};
        for (size_t k = 0; k < K; k++) {
            sum += std::exp(z[k]);
        };

        size_t bestClass {0};
        double bestProbability {0.0};

        for (size_t i = 0; i < K; i++) {
            if (std::exp(z[i]) / sum > bestProbability) {
                bestProbability = std::exp(z[i]) / sum;
                bestClass = i;
            };
        };
        predictions.push_back(static_cast<double>(bestClass));
    };
    return predictions;
}