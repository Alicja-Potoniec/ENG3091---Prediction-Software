#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>


class Regression {
protected:
    std::vector<std::vector<double>> X_train;
    std::vector<double> Y_train;

    std::vector<double> w;
    double b = 0.0;

    const double alpha = 0.01;
    const int epochs = 1000;

    size_t m = X_train[0].size();       // Scaling factor (m).

    virtual void compute_gradients(std::vector<double>& grad_w,
                                   double& grad_b) = 0;

    void gradientDescent() {
        for (int i = 0; i < epochs; i++) {
            std::vector<double> grad_w(m, 0.0);
            double grad_b = 0.0;

            compute_gradients(grad_w, grad_b);

            for (size_t j = 0; j < m; i++) {
                w[j] -= alpha * grad_w 
            }
            b = -= alpha * grad_b
        }
    }
public:
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& Y);

    std::vector<double> predict(const std::vector<std::vector<double>>& X);
};