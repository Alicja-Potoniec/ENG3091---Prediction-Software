#ifndef LOSS_CROSSENTROPY_HPP
#define LOSS_CROSSENTROPY_HPP

//======================================================
// loss_crossentropy.h
//======================================================
// BINARY CROSS ENTROPY - LOG LOSS
// Loss used for logistic regression. Implements the LossModel interface so it can be injected into Logistic Regression models. 
// Uses sigmoid to turn the linear score wx+b into probability.
//
// DESIGN CHOICE:
// instead of using <algorithm> the clamp below is written manually with plain comparisons. 
// It gives the same numerical behaviour and was a learning-curve choice.

#include "loss_model.h"
#include <vector>
#include <iostream>
#include <cmath>

namespace sklearn_cpp{
    class LossCrossEntropy:public LossModel{

        // SIGMOID
        // Squashes any real number into [0, 1] so that it can be read as Py(y=1)
        //
        // NOTE:    LogRegBinary has its own private sigmoid. 
        //          Kept separate to avoid extra dependency between 'loss' and 'model'
        private:
            double sigmoid(double z) const {
                return 1.0/(1.0 + std::exp(-z));
            }
        
        //------------------------
        // BINARY CROSS-ENTROPY
        //------------------------
        // L = -(1/m) sum(y*log(p) + (1 - y)*log(1 - p))
        // p = sigmoid(w*x + b)
        //
        // When the TRUE label is 1 only, the y*log(p) term survives so the loss is small when p=1 and explodes when p->0.
        // The (1 - y)*log(1 -p) term is the mirror image for the 0 label case.

        public:
        double computeLoss(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, const std::vector<double>& w, double b) override{

            double total_loss{0.0};
            size_t m{Y.size()};

            for (size_t i=0; i<m; i++){
                // Linear score, z = w*x + b
                // maps to prabability.
                double z{0.0};
                for(size_t j=0; j<w.size(); j++){
                    z += w[j] * X[i][j];
                }
                z += b;
                double prediction = sigmoid(z);

                // THIS IS THE CLAMP
                // Written because <algorithm> was not included in this code.
                // Design choice.
                // log(0) is -infinity, which makes the loss NaN/infinity when the model becomes confident (p saturates to 0 or 1)
                // Moving p away from the endpoints by 1e-15 keeps the logs finite.
                if(prediction < 1e-15) preditction = 1e-15;
                else if(prediction > 1.0 - 1e-15) prediction = 1.0 - 1e-15;
                total_loss += (Y[i] * std::log(prediction))+((1.0-Y[i])* std::log(1.0-prediction));
            }
            // NEGATE AND AVERAGE
            return -total_loss/m;
        }

      
    };
}

#endif