#ifndef SOFTMAX_REGRESSION_HPP
#define SOFTMAX_REGRESSION_HPP

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <algorithm>

#include "LogReg_binary.h"
#include "loss_model.h"

namespace sklearn_cpp {
    namespace linear_model {
        
        // ====================================
        // Tool 2 == MULTICLASS LOGISTIC REGRESSION 
        // ==================================== 
        // Inherit member functions and variables from LogRegBinary in order to handle K classes using softmax function.
        // Each class has its own weight vector and bias, stored in W_multi and B_multi. The class with the highest probability 
        // is the prediction.
        class LogRegSoftmax : public LogRegBinary {
            private:
                // Number of target classes (defined at construction and fixed for the lifetime of the model)
                int K{0};

                // Per class weight matrix and bias vector.
                // Basically W_multi[k] holds the weight vector for class k while,
                // B_multi[k] holds the bias for class k
                std::vector<std::vector<double>> W_multi;
                std::vector<double> B_multi;

                // === SOFTMAX CALCULATION ===
                // Converts a vector of raw class scores into a probability distribution.
                // Each output is in the range (0, 1), and all outputs sum to 1.
                // max_z is subtracted from each score before exponentiation to provide numerically stable formulation
                // preventing std::exp overflow for large inputs while producing identical results mathematically.
                std::vector<double> softmax(const std::vector<double>& z) const {
                    std::vector<double> prob(z.size());

                    // Finding the max value
                    // can be replaced with a single line with double max_z = *std::max_element(z.begin(), z.end());
                    // To do this the header would need to #include <algorithm>
                    double max_z{z[0]};
                    for(size_t i=1; i<z.size(); i++){
                        if(z[i]>max_z) max_z = z[i];
                    }

                    // Calculate the normalisation denominator
                    double sum {0.0};
                    for (double val : z) {
                        sum += std::exp(val - max_z);
                    }

                    // Divide the shifted exponentials by the sum to produce the probabilities
                    for (size_t i = 0; i < z.size(); i++) {
                        prob[i] = std::exp(z[i] - max_z) / sum;
                    }

                return prob;
                }
            
            protected:
                // === NECESSARY OVERRIDE ====
                // compute_gradients is a pure virtual function, so we must ovveride it in every concrete subclass.
                // Softmax has its own seperate gradient function, so this is empty and only done to satisfy the compiler without 
                // interfering with training.
                void compute_gradients(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, std::vector<double>& grad_w, double& grad_b) override {}

                // === SOFTMAX GRADIENT COMPUTATION ===
                // Calculates the mean gradient of the cross-entropy loss w.r.t all k weight vectors
                // and biases. The gradient for each class k is simplified to 
                // prob[k] - 1(y[i] == k), where 1(y[i] == k) is the indicator function : 1 is k is the true class and 0 otherwise.
                void compute_softmax_gradients(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, std::vector<std::vector<double>>& grad_w, std::vector<double>& grad_b) {
                    for (size_t i = 0; i < X.size(); i++) {
                        std::vector<double> z (K,0.0);

                        // Calculate the vector z for sample i across all classes
                        for (int k = 0; k < K; k++) {
                            z[k] = B_multi[k];
                            for (size_t j = 0; j < X[i].size(); j++) {
                                z[k] += W_multi[k][j] * X[i][j];                     
                            }
                        }

                        // Define probabilities
                        std::vector<double> prob {softmax(z)};

                        for (int k = 0; k < K; k++) {
                            // Define error as indicator function 1(y[i] == k) subtracted from predicted probability
                            double error = prob[k] - (Y[i] == static_cast<double>(k) ? 1.0 :0.0);

                            // Chain rule: Scale the error by each input feature
                            for (size_t j = 0; j < X[i].size(); j++) {
                                grad_w[k][j] += error * X[i][j];
                            }

                            grad_b[k] += error;
                        }
                    }
                    
                    // Normalise the gradients by batch size
                    double scale = 1.0 / X.size();

                    for (int k = 0; k < K; k++) {
                        for (size_t j = 0; j < W_multi[k].size(); j++) {
                            grad_w[k][j] *= scale;
                        }

                        grad_b[k] *= scale;
                    }
                }
                
            public: 
                // Constructor for LogRegBinary parameter initialisation, then resize W_multi and B_multi to hold one weight vector and bias per class.
                // All weights and biases are initialised to zero
                LogRegSoftmax(int n_features,int K, LossModel& loss, double lr = 0.01, int iter = 1000): LogRegBinary(n_features, loss, lr, iter), K{K} {
                    W_multi.resize(K, std::vector<double>(n_features, 0.0));
                    B_multi.resize(K,0.0);
                }
                
                // === FIT ===
                // Override the BaseModel fit in order to use softmax specific gradient function.
                // Gradients freshly initialised each iteration to prevent contaminationg from previous gradient iterations
                void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) {
                    for (int i = 0; i < iterations; i++) {
                        // Fresh gradient accumulator for this iteration
                        std::vector<std::vector<double>> grad_w(K, std::vector<double>(W_multi[0].size(), 0.0));
                        std::vector<double> grad_b(K, 0.0);

                        compute_softmax_gradients(X, Y, grad_w, grad_b);

                        // Subtract the scaled gradient from each parameter
                        for (int k = 0; k < K; k++) {
                            for (size_t j = 0; j < W_multi[k].size(); j++) {
                                W_multi[k][j] -= learning_rate * grad_w[k][j];
                            }
                            B_multi[k] -= learning_rate * grad_b[k];
                        }

                        // Progress indicator that overwrites current line to avoid flooding console
                        // This progress indicator is printed every 100 iterations so long-running training 
                        // on high-dimensional data remains observable.
                        if (i % 100 == 0) {
                            std::cout << "Progress:" << (static_cast<double>(i) / iterations) * 100 << "%\r" << std::flush;
                        }           
                    }
                }
                
                // === PREDICT ===
                // Calculate the softmax probability distribution for each sample and
                // returns the index of the class with the highest probability as the predicted class.
                // Heap memory is allocated to predictions through reserve() to prevent heap reallocation
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
                        
                        // Scan to find the class with the highest probability
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

                // === PRINT LOSS ===
                // Performs cross-entropy loss calculation over the dataset.
                // Loss only makes use of the log probability of the true class only,
                // the other classes make no contribution for a given sample.
                // A clamp of 1e-15 guards against log(0) when a probability saturates to zero. 
                void printLoss(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) const {

                    double loss {0.0};
                    int correct {0};
                    for (size_t i = 0; i < X.size(); i++) {
                        std::vector<double> z(K,0.0);

                        for (int k = 0; k < K; k++) {
                            z[k] = B_multi[k];
                            for (size_t j = 0; j < X[i].size(); j++) {
                                z[k] += W_multi[k][j] * X[i][j];
                            };
                        };

                        //Define probabilities
                        std::vector<double> prob {softmax(z)};
                        
                        // Determine the true class index and find its log probability
                        int true_class {static_cast<int>(Y[i])};
                        loss += -std::log(std::max(prob[true_class], 1e-15));

                        // The predicted class is the index of the highest probability
                        int predicted = std::max_element(prob.begin(), prob.end()) - prob.begin();
                        if (predicted == true_class) {
                            correct++;
                        }
                    }
                    std::cout << std::fixed << std::setprecision(4);

                    std::cout << "=====Multiclass Logistic (Softmax) REGRESSION RESULTS=====\n";
                    for (int k = 0; k < K; k++) {
                        std::cout << "Class " << k << ": b = " << B_multi[k] << "\n";
                    }
                    // Loss is normalised by the number of samples to report the mean cross-entropy
                    std::cout << "Loss: " << loss / X.size() << "\n";
                    std::cout << "Accuracy: " << (static_cast<double>(correct) / X.size()) * 100 << "%\n";
                }
        };
    }
}


#endif