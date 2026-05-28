#ifndef SOFTMAX_REGRESSION_HPP
#define SOFTMAX_REGRESSION_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

#include "base_model.h"
#include "loss_model.h"

namespace sklearn_cpp {
    namespace linear_model {
        
        class LogRegSoftmax : public BaseModel {
            private:
                LossModel& loss_model;

                int K;
                std::vector<std::vector<double>> W_multi;
                std::vector<double> B_multi;

                std::vector<double> softmax(const std::vector<double>& z) const {
                    std::vector<double> prob(z.size());

                    // Finding the max value
                    // can be replaced with a single line with double max_z = *std::max_element(z.begin(), z.end());
                    // To do this the header would need to #include <algorithm>
                    double max_z{z[0]};
                    for(size_t i=1; i<z.size(); i++){
                        if(z[i]>max_z) max_z = z[i];
                    }

                    double sum {0.0};
                    for (double val : z) {
                        sum += std::exp(val - max_z);
                    }

                    for (size_t i = 0; i < z.size(); i++) {
                        prob[i] = std::exp(z[i] - max_z) / sum;
                    }

                return prob;
                }
            
            protected:
                void compute_gradients(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, std::vector<double>& grad_w, double& grad_b) override {}

                void compute_softmax_gradients(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, std::vector<std::vector<double>>& grad_w, std::vector<double>& grad_b) {
                    for (size_t i = 0; i < X.size(); i++) {
                        std::vector<double> z (K,0.0);

                        for (int k = 0; k < K; k++) {
                            z[k] = B_multi[k];
                            for (size_t j = 0; j < X[i].size(); j++) {
                                z[k] += W_multi[k][j] * X[i][j];                     
                            }
                        }

                        std::vector<double> prob {softmax(z)};

                        for (int k = 0; k < K; k++) {
                            double error = prob[k] - (Y[i] == static_cast<double>(k) ? 1.0 :0.0);
                            for (size_t j = 0; j < X[i].size(); j++) {
                                grad_w[k][j] += error * X[i][j];
                            }

                            grad_b[k] += error;
                        }
                    }
                
                    double scale = 1.0 / X.size();

                    for (int k = 0; k < K; k++) {
                        for (size_t j = 0; j < W_multi[k].size(); j++) {
                            grad_w[k][j] *= scale;
                        }

                        grad_b[k] *= scale;
                    }
                }
                
            public: 
                LogRegSoftmax(int n_features,int K, LossModel& loss, double lr = 0.01, int iter = 1000): BaseModel{n_features, lr, iter}, loss_model{loss}, K{K} {
                    W_multi.resize(K, std::vector<double>(n_features, 0.0));
                    B_multi.resize(K,0.0);
                }
                
                void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
                    for (int i = 0; i < iterations; i++) {
                        std::vector<std::vector<double>> grad_w(K, std::vector<double>(W_multi[0].size(), 0.0));
                        std::vector<double> grad_b(K, 0.0);

                        compute_softmax_gradients(X, Y, grad_w, grad_b);

                        for (int k = 0; k < K; k++) {
                            for (size_t j = 0; j < W_multi[k].size(); j++) {
                                W_multi[k][j] -= learning_rate * grad_w[k][j];
                            }
                            B_multi[k] -= learning_rate * grad_b[k];
                        }
                    }
                }

                std::vector<double> predict(const std::vector<std::vector<double>>& X) const override {

                    std::vector<double> predictions;
                    predictions.reserve(X.size());

                    for (size_t i = 0; i < X.size(); i++) {

                        std::vector<double> z (K, 0.0);

                        for (int k = 0; k < K; k++) {
                            z[k] = B_multi[k];
                            for (size_t j = 0; j < X[i].size(); j++) {
                                z[k] += W_multi[k][j] * X[i][j];
                            }
                        }
                        
                        std::vector<double> prob = softmax(z);

                        int bestClass {0};
                        double bestProbability {prob[0]};

                        for (int k = 1; k < K; k++) {
                            if (prob[k] > bestProbability) {
                                bestProbability = prob[k];
                                bestClass = k;
                            }
                        }
                        
                        predictions.push_back(bestClass);
                    }
                    return predictions;
                }

                void printLoss(const std::vector<std::vector<double>>& X, std::vector<double>& Y) const {

                    std::cout << std::fixed << std::setprecision(4);

                    std::cout << "=====Multiclass Logistic (Softmax) REGRESSION RESULTS=====\n";
                    for (int k = 0; k < K; k++) {
                        std::cout << "Class " << k << ":\n";
                        std::cout << "b = " << B_multi[k] << "\n";
                    }
                    std::cout << "Loss: " << loss_model.computeLoss(X, Y, W_multi[0], B_multi[0]) << "\n";
                }
        };
    }
}


#endif