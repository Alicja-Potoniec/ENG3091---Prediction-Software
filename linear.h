#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <iostream>
#include <iomanip>
#include <vector>

#include "base_model.h"
#include "loss_model.h"

namespace sklearn_cpp{

    class LinearRegression:public BaseModel{
        protected:
        void compute_gradients(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, std::vector<double> grad_w, double& grad_b)override{

            size_t m{Y.size()};
            for(size_t i=0; i<m; i++){
                double y_hat{b};
                for(size_t j=0; j<w.size(); j++){
                    y_hat += w[j] * X[i][j];
                }
                double residual = y_hat - Y[i];
                for(size_t j=0; j<w.size(); j++){
                    grad_w[j] += residual * X[i][j];
                }
                grad_b += residual;
            }
            double scale = 2.0/static_cast<double>(m);
            for(size_t j=0; j<w.size(); j++){
                grad_w[j] *= scale;
            }
            grad_b *= scale;
        }
        private: 
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
        public:
        LinearRegression(int n_features, LossModel& loss, double lr=0.01, int iter=1000):BaseModel(n_features, lr, iter)loss_model(loss){}

        std::vector<double>predict(const std::vector<std::vector<double>>& X)const override{
            std::vector<double>predictions;
            predictions.reserve(X.size());

            for(size_t i=0; i<X.size(); i++){
                double y_hat{b};
                for(size_t j=0; j<w.size(); j++){
                    y_hat += w[j] * X[i][j];
                }
                predictions.push_back(y_hat);
            }
            return predictions;
        }
        void printLoss(const std::vector<std::vector<double>>& X, const std::vector<double>& Y)const{

            double mse = loss_model.computeLoss(X, Y, w, b);
            double r2 = coputeR2(X, Y);

            std::cout << std::fixed << std::setprecision(4);
            std::cout << "=====LINEAR REGRESSION RESULTS=====\n";
            std::cout << "MSE Loss:" << mse << "\n";
            std::cout << "R2 score:" << r2 << "\n";
        }
        void printPara()const{
            
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "=====LEARNED PARAMETERS=====";
            for(size_t j=0; j<w.size(); j++){
                std::cout << "w[" << j << "] =" << w[j] << "\n";
            }
            std::cout << "b =" << b << "\n";
        }
    };
}

#endif