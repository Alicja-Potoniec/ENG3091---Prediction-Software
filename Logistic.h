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

        // ====================================
        // Tool 1 == BINARY LOGISTIC REGRESSION 
        // ==================================== 
        // Inherits gradient descent and parameter storage from Base model.
        // Uses sigmoid computation and makes it binary cross-entropy compatible
        // gradient computation for two class classification problems (label 0 or 1)
        class LogRegBinary : public BaseModel {
            
            // Data members which are hidden from the user interface
            protected:
                // Injected at construction time to allow different loss functions to be used without modifying this class
                LossModel& loss_model;

                // === SIGMOID COMPUTATION ===
                // Takes z as a const reference to avoid copying the scalar and improve runtime
                // Maps the real value input z to the range (0, 1).
                // This is basically the probability that the sample belongs to class 1
                double sigmoid(const double& z) const {
                    return 1.0 / (1.0 + std::exp(-z));
                }
                
                // === ACCURACY ===
                // Evaluates classification performance through hard prediction comparison against
                // ground truth labels. Sigmoid outputs above 0.5 are classified as 1 and below this they are zero.
                // This is seperate from printLoss so it can be reused independently 
                double Accuracy(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) const {
                    size_t correct{0};
                    for (size_t i = 0; i < Y.size(); i++) {
                        // Calculate z = w * x + b for sample i
                        double z{b};
                        for (size_t j = 0; j < w.size(); j++) {
                            z += w[j] * X[i][j];
                        }
                        double prediction = (sigmoid(z) >= 0.5) ? 1.0 : 0.0;
                        if (prediction == Y[i]) correct++;
                    }
                    // Return proportion correct as a value between 0 and 1
                    return static_cast<double>(correct) / Y.size();
                }

                // === GRADIENT DESCENT CALCULATION ===
                // Override pure virtual function in BaseModel.
                // Calculate the mean gradient of the binar cross-entropy loss w.r.t all
                // weights and bias across the training batch. The gradient is simplified to
                // sigmoid(z) - y when differentiating log loss through the sigmoid.
                void compute_gradients(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, std::vector<double>& grad_w, double& grad_b) override {
                    size_t m{Y.size()};
                    for (size_t i = 0; i < m; i++) {
                        // Accumulate the z for sample i
                        double z{b};
                        for (size_t j = 0; j < w.size(); j++) {
                            z += w[j] * X[i][j];
                        }
                        // Define error as the true label subtracted from the predicted probability
                        // Positive error means the model has over predicted
                        // Negative error means the model under predicted
                        double error = sigmoid(z) - Y[i];

                        // Chain rule: Scale the error by each input feature to get the contribution of 
                        // each weight to the total loss
                        for (size_t j = 0; j < w.size(); j++) {
                            grad_w[j] += error * X[i][j];
                        }
                        grad_b += error;
                    }
                    // Normalise the gradients by batch size
                    double scale = 1.0 / static_cast<double>(m);
                    for (size_t j = 0; j < w.size(); j++) {
                        grad_w[j] *= scale;
                    }
                    grad_b *= scale;
                }

            public:
                // Constructor for BaseModel parameter initialisation and bind loss function to member reference.
                LogRegBinary(int n_features, LossModel& loss, double lr = 0.01, int iter = 1000)
                    : BaseModel(n_features, lr, iter), loss_model(loss) {}
                
                // === PREDICT ===
                // Uses learned parameters to produce class labels (0 or 1)
                // Heap memory is allocated to predictions through reserve() to prevent repeated 
                // allocation when pushing predictions into vector
                std::vector<double> predict(const std::vector<std::vector<double>>& X) const override {
                    std::vector<double> predictions;
                    predictions.reserve(X.size());
                    for (size_t i = 0; i < X.size(); i++) {
                        double z{b};
                        for (size_t j = 0; j < w.size(); j++) {
                            z += w[j] * X[i][j];
                        }
                        // 0.5 threshold to convert probability to binary label
                        predictions.push_back(sigmoid(z) >= 0.5 ? 1.0 : 0.0);
                    }
                    return predictions;
                }

                // === PREDICT PROBABILITY ===
                // Returns the actual sigmoid output rather than a class label.
                // Used to inspect model confidence.
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

                // === PRINT LOSS ===
                // Performs loss calculation using the injected LossModel
                // making the evaluation process decoupled from this class.
                // Accuracy is found using a member function and reported alongside the loss
                // to showcase model preformance. 
                void printLoss(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) const {
                    double loss = loss_model.computeLoss(X, Y, w, b);
                    double accuracy = Accuracy(X, Y);
                    std::cout << std::fixed << std::setprecision(4);
                    std::cout << "=====Logistic REGRESSION RESULTS=====\n";
                    std::cout << "Binary Loss: " << loss << "\n";
                    std::cout << "Accuracy: " << accuracy * 100.0 << "%\n";
                }

                // === PRINT PARAMETERS ===
                // Prints learned weights and the bias after training
                // Enables inspection of features which the model weighted most heavily
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