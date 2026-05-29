#ifndef LOSS_MSE_HPP
#define LOSS_MSE_HPP

//=====================================
// loss_MSE.h
//=====================================
// MEAN SQUARED ERROR
// Loss used for linear regression.
// Implements the LossModel so it can be injected like any other loss function
// It does provide computeR2(), the coefficeint for determination which is the standard goodness of fit score for regressors.

#include "loss_model.h"
#include <iostream>
#include <vector> 

namespace sklearn_cpp{

    class LossMSE: public LossModel{

        //----------------------------------
        // MEAN SQUARED ERROR
        //----------------------------------
        // WHY the (1/2m) instead of (1/m)  - the factor of 2 in the denominator cancels the 2 that drops our after differentiating the squared term, resulting in a clean gradient
        // This does not change where the minimum is placed, however, it scales the loss value. 
        // 
        // override     - forced the compiler to verify if this matches the LossModel signature. Cheap fix against silent signature drift.
        public:
        double computeLoss(const std::vector<std::vector<double>>& X, const std::vector<double>&Y, const std::vector<double>&w, double b) override {

            double total_squared_error{0.0};
            size_t m{Y.size()};

            for (size_t i=0; i<m; i++){

                // LINEAR PREDICTION FOR SAMPLE:    i: y_hat = w*x + b
                // Computed with an explicit dot product loop isntead of a matrix library
                // This is to keep the project dependancy-free and the maths visible. 
                //
                // BLAS/ Eigen dot product would be the alternative on large data

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
        //---------------------------------------------------
        // R^2 COEFFICIENT OF DETERMINATION
        //---------------------------------------------------
        //R^2 = 1 - SS_res / SS_tot
        //
        // SS_res       - squared error of the model
        // SS_tot       - total squared error of predicting the mean of Y
        //
        // R^2 = 1      - perfect
        // R^2 = 0      - no better that the mean
        // this number allows to evaluate the effectiveness of the fit
        //
        // KNOWN DUPLICATION
        // See note in linear.h
        // Deliberate trade-off
        // Linearregression has its own duplicate computeR2 function. This logic lives in two places. 
        // LinearRegression could have delegated to this methos as it already holds the LossModel& interface.
        // However, it would need a downcast or an R2 method on the interface itself. 
        // This way R2 on the LossModel stays MSE specific instead of leaking into the abstract base that classification losses also share.
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