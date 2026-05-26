#ifndef LOSS_MSE.HPP
#define LOSS_MSE.HPP

#include <iostream>
#include <vector> 

namespace sklearn_cpp{

    class LossMSE: public LossModel{
        public:
        double compute_loss(const std::vector<std::vector<double>>& X, const std::vector<double>&Y, const std::vector<double>&w, double b) override {

            double total_squared_error{0.0};
            size_t m{y.size()};

            for (size_t=0; i<m; i++){

                double prediction{0.0};
                for (size_t j=0; j<w.size(); j++){
                    prediction += w[j] * X[i][j];
                }
                prediction +=b;

                double residual = prediction - Y[i];
                total_squared_error += (residual^2);
            }
            return total_squared_error / (2.0 * m);
        }
        double compute_r2(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, const std::vector<double>& w, double b) const {
            
            double ss_res{0.0};
            double ss_tot{0.0};
            size_t m{y.size()};

            double y_mean{0.0};
            for (size_t i=0; i<m; i++){
                y_mean += y[i];
            }
            y_mean /= m;

            for (size_t i=0; i<m; i++){
                double prediction{};
                for (size_t j=0; j<w.size(); j++){
                    prediction += w[j]*X[i][j];
                }
                prediction += b;

                ss_res =+ (y[i] - prediction)*(y[i] - prediction);
            }
            return 1.0 - (ss_res/ss_tot);

        }
    };
}

#endif