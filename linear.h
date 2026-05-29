#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

//====================================================
// linear.h
//====================================================
// Linear Regression - Predict a continuous value as y_hat=w*x + c
// Inherits gradient descent machinery and parameter storage from BaseModel
// Only supplies the MSE gradient and the prediction routine. 

#include <iostream>
#include <iomanip>
#include <vector>
#include "base_model.h"
#include "loss_model.h"

namespace sklearn_cpp {
    namespace linear_model {

    class LinearRegression : public BaseModel {

        // DESIGN CHOICE
        // loss_model is held as a REFERENCE to the abstract LossModel
        // This is the place where owning a smart pointer, std::unique_ptr<LossModel>, was a genuine alternative.
        // This is because storing the polymorphic base-class handles were only known at the call site. 
        // REASON:      Using a reference improves the ease of use of the program within main.cpp
        //              Ownership stays with the caller. In main.cpp the loss is an ordinary object that outlives the model. There is nothing to allocate/transfer or free.
        //              Zero heap allocation, zero ownership, nothing to delete
        //              One loss for the whole life of the model.
        //TRADE OFF:    Because this is a reference, the CALLER MUST keep the object alive for as long as the model is in use. 
        //              Everything lives in main.cpp for the duration of training. Hence the lighter, simpler reference was prefered over the heavier owning pointer.
        private:
            LossModel& loss_model;   

            //-------------------------------
            // R^2 SCORE
            //-------------------------------
            // the goodness-of-fit
            // Reported by printLoss()
            // SEE NOTE in loss_MSE.h: LossMSE also has compute R2, this logic is intentionally duplicated. 
            // This is to make the program versatile if there was ever a wish to expand it. 
            // A private copy is included here so that it can score itself directly from its own w/b without downcadting the LossModel& back to LossMSE&
            double computeR2(const std::vector<std::vector<double>>& X, const std::vector<double>& Y) const {  

                double ss_res {0.0};
                double ss_tot {0.0};
                size_t m {Y.size()};

                double Y_mean {0.0};
                for (size_t i = 0; i < m; i++) Y_mean += Y[i];
                Y_mean /= m;

                for (size_t i = 0; i < m; i++) {
                    double prediction {b};
                    for (size_t j = 0; j < w.size(); j++) {
                        prediction += w[j] * X[i][j];
                    }
                    ss_res += (Y[i] - prediction) * (Y[i] - prediction);
                    ss_tot += (Y[i] - Y_mean) * (Y[i] - Y_mean);
                }
                return 1.0 - (ss_res / ss_tot);
            }

        // ---------------------------------------------------
        // MSE GRADIENT
        // Gradient of mean squared error w.r.t. each weight and the bias
        // dL/dw_j = (2/m) sum((y_hat - y)*x_j)
        // dL/db = (2/m) sum(y_hat - y)
        // Sums are accumulated first, then 2/m scale is applied once at the end (cheaper than scaling inside of the loop)
        //
        // NOTE - CONSTANT FACTOR
        // The reported loss in loss_MSE.h uses the (1/2m) convention, the exacr derivative for this is: (1/m)sum(r*x)
        // The extra constant factor of 2 rescales only the effective step size, it doesn't change the direction of descent or the final fit, it's absorbed into learning rate. 
        // Training is identical up to that rescaling.
        protected:
            void compute_gradients(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, std::vector<double>& grad_w, double& grad_b) override {

                size_t m {Y.size()};
                for (size_t i = 0; i < m; i++) {
                    double y_hat {b};
                    for (size_t j = 0; j < w.size(); j++) {
                        y_hat += w[j] * X[i][j];
                    }
                    double residual = y_hat - Y[i];
                    for (size_t j = 0; j < w.size(); j++) {
                        grad_w[j] += residual * X[i][j];
                    }
                    grad_b += residual;
                }
                double scale = 2.0 / static_cast<double>(m);
                for (size_t j = 0; j < w.size(); j++) {
                    grad_w[j] *= scale;
                }
                grad_b *= scale;
            }

        // Forwards the model parameters to BaseModel and binds the loss reference. 
        // Member-initialiser list is because the reference number (loss_model) MUST be initialised at construction, cannot be assigned later.
        public:
            LinearRegression(int n_features, LossModel& loss, double lr = 0.01, int iter = 1000): BaseModel(n_features, lr, iter), loss_model(loss) {}

            //---------------------------------
            // PREDICT
            //---------------------------------
            // FOR LINEAR REGRESSION THE PREDICTION IS THE LINEAR SCORE
            // y_hat = w*x + b
            // No thresholding.
            // reserve() pre-sizes the output vector so the push_back loop never reallocates. A win for efficiency.
            std::vector<double> predict(const std::vector<std::vector<double>>& X) const override {

                std::vector<double> predictions;
                predictions.reserve(X.size());
                for (size_t i = 0; i < X.size(); i++) {
                    double y_hat {b};
                    for (size_t j = 0; j < w.size(); j++) {
                        y_hat += w[j] * X[i][j];
                    }
                    predictions.push_back(y_hat);
                }
                return predictions;
            }

            //----------------------------
            // REPORTING
            //----------------------------
            // printLoss delegates the MSE number to the injected loss object (decoupling the score from this class)
            // Pairs it with R^2 so the reader gets both the raw error and the normalised fit quality.
            // std::fixed + setprecision(4) - gives stable readable decimals. 
            void printLoss(
                const std::vector<std::vector<double>>& X, const std::vector<double>& Y) const {

                double mse = loss_model.computeLoss(X, Y, w, b);
                double r2  = computeR2(X, Y);  

                std::cout << std::fixed << std::setprecision(4);    // sets result to 4 d.p.
                std::cout << "=====LINEAR REGRESSION RESULTS=====\n";
                std::cout << "MSE Loss: " << mse << "\n";
                std::cout << "R2 Score: " << r2  << "\n";
            }

            // Dumps the learned weights and biases
            // useful for inspecting which features the model leaned on, however, quite redundant for large datasets.
            // Was used initially when testing the models, but unnecessary when wanting to obtain the pure result.
            void printPara() const {
                std::cout << std::fixed << std::setprecision(6);
                std::cout << "=====LEARNED PARAMETERS=====\n";
                for (size_t j = 0; j < w.size(); j++) {
                    std::cout << "w[" << j << "] = " << w[j] << "\n";
                }
                std::cout << "b = " << b << "\n";
            }
    };

    } 
} 

#endif