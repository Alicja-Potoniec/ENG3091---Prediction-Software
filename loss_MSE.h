#ifndef LOSS_MSE_HPP
#define LOSS_MSE_HPP

#include "loss_model.hpp"
#include <iostream>
#include <vector> 

namespace sklearn_cpp{

    class LossMSE: public LossModel{
        public:
        double computeLoss(const std::vector<std::vector<double>>& X, const std::vector<double>&Y, const std::vector<double>&w, double b) override {

            double total_squared_error{0.0};
            size_t m{Y.size()};

            for (size_t i=0; i<m; i++){

                double prediction{0.0};
                for (size_t j=0; j<w.size(); j++){
                    prediction += w[j] * X[i][j];
                }
                prediction +=b;

                double residual = prediction - Y[i];
                total_squared_error += (residual * residual);
            }
            return total_squared_error / (2.0 * m);
        }
        double computeR2(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, const std::vector<double>& w, double b) const {
            
            double ss_res{0.0};
            double ss_tot{0.0};
            size_t m{Y.size()};

            double Y_mean{0.0};
            for (size_t i=0; i<m; i++){
                Y_mean += Y[i];
            }
            Y_mean /= m;

            for (size_t i=0; i<m; i++){
                double prediction{};
                for (size_t j=0; j<w.size(); j++){
                    prediction += w[j]*X[i][j];
                }
                prediction += b;

                ss_res += (Y[i] - prediction)*(Y[i] - prediction);
                ss_tot += (Y[i] - Y_mean)*(Y[i] - Y_mean);
            }
            return 1.0 - (ss_res/ss_tot);

        }
    };
}

#endif