#ifndef LOSS_CROSSENTROPY_HPP
#define LOSS_CROSSENTROPY_HPP

#include "loss_model.h"
#include <vector>
#include <iostream>
#include <cmath>

namespace sklearn_cpp{
    class LossCrossentropy:public LossModel{

        private:
            double sigmoid(double z){
                return 1.0/(1.0 + std::exp(-z));
            }

        public:
        double computeLoss(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, const std::vector<double>& w, double b)override{

            double total_loss{0.0};
            size_t m={Y.size()};

            for (size_t i=0; i<m; i++){
                double z{0.0};
                for(size_t j=0; j<w.size(); j++){
                    z += w[j] * X[i][j];
                }
                z += b;
                double prediction = sigmoid(z);
                total_loss += (Y[i] * std::log(prediction))+((1.0-Y[i])* std::log(1.0-prediction));
            }
            return -total_loss/m;
        }

      
    };
}

#endif