#ifndef BASEMODEL_HPP
#define BASEMODEL_HPP

// ==========================================
// base_model.h
//===========================================
// ABSTRACT BASE
// Shared by linear.h & LogReg_binary.h (THROUGH LogRegBinary its shared into LogReg_multiclass.h)
// -------------
// OWNS: 
// w            - Weight vector
// b            - Bias
// lr           - learning rate
// iterations   - iteration count
// gradient descent function
//
// This is the "fixed" algorithm. Zero the gradient accumulators asks the subclass to fill them, takes a step then repeat.
// Subclass specific step is the pure virtual hook - compute_gradients()
// Base class controls the flow
// Derived class fills in the varying detail

#include <iostream>
#include <vector>

namespace sklearn_cpp{

    class BaseModel{

        // PROTECTED CLASS - Design choice so that the subclasses can read/write these directly into their own gradient and predition code.
        // whilst still staying hidden from outside callers, these only ever see fit() or predict().
        protected:
        std::vector<double> w;  // one weight per feature
        double b{0.0};          // single bias/ intercept term
        double learning_rate;   // step size for each gradient-descent update
        int iterations;         // number of full phases over the data

        //-------------------------
        // GRADIENT DESCENT
        //-------------------------
        // Repeats 'iterations' times:
        // Builds fresh zeroed accumulators, lets the subclass fill them via the compute_gradients()
        // Afterwards, the function nudges each parameter a small step downhill (by using the -= call)
        //
        // Every iteration uses the entire dataset.

        void gradient_descent(const std::vector<std::vector<double>>& X, const std::vector<double>& Y){
            size_t n{w.size()};
            for (int iter=0; iter<iterations; iter++){

                std::vector<double>grad_w(n, 0.0);      // ACCUMULATORS
                                                        // so gradients after each iteration never leak from the previous step into the next one.
                double grad_b{0.0};
                // Filled in by the subclass.
                compute_gradients(X, Y, grad_w, grad_b);
                for (size_t j=0; j<n; j++){
                    w[j] -= learning_rate*grad_w[j];
                }
                b -= learning_rate*grad_b;
            }
        }
        // ---------------------------------------------------------------------
        // MUST BE PROVIDED BY THE SUBCLASS
        // ---------------------------------------------------------------------
        // Pure virtual (=0). There is no sensible generic gradeint. 
        // The base is not initialised and forces the model to determine this. 
        // grad_w & grad_b are output parameters. - The base class populates them with its output. 
        // The rest is completed by gradient_descent()
        virtual void compute_gradients(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, std::vector<double>& grad_w, double& grad_b) =0;
        
        //---------------------------------------------------------------------
        // INPUT VALIDATION
        //---------------------------------------------------------------------
        // Shape checks to be completed before running:
        // Check if X & Y rows are matching
        // Check if feature count matches the weight vector
        // CONST - this function is INSPECT ONLY.
        // Reports issues instead of breaking or printing garbage.
        private:
        bool validate(const std::vector<std::vector<double>>& X, const std::vector<double>& Y)const{
            if(X.empty() || Y.empty()){
                std::cerr<<"Error: Missing data. \n";
                return false;
            }
        if(X.size()!= Y.size()){
            std::cerr<<"Error: X & Y row counts do not match. \n";
            return false;
        }    
        if (X[0].size() != w.size()){
            std::cerr<<"Error: Feature counts do not match. \n";
            return false;
        }
        return true;
        }
        // ==============================
        // CONSTRUCTOR
        // ==============================
        // POLYMORPHIC
        // Sizes the weight vector to n_features and performs ZERO-INITIALISATION for every weight and bias. 
        // Gradient descent moves the weights to a good solution regardless of the zero start.
        // Sufficient for a single linear layer.
        // lr and iter have defaults so that the model can be built with just the feature count.
        public:
        BaseModel(int n_features, double lr=0.01, int iter=1000): w (n_features, 0.0), b(0.0), learning_rate(lr), iterations (iter){}

        // VIRTUAL DESTRUCTOR - to prevent memory bottleneck
        virtual ~BaseModel() = default;
        
        //-------------------
        // FIT
        // ------------------
        // PUBLIC TRAINING
        // VALIDATE FIRST -> RUN TO DESCENT
        // NON-VIRTUAL - so that every model can share the check function. 
        // MULTICLASS MODEL IS THE EXCEPTION - due to the different parameter layout. 
        void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y){
            if (!validate(X, Y)) return;
            gradient_descent(X, Y);
        }
        // PURE VIRTUAL
        // Every model MUST say how to turn inputs into preditions
        // CONST becaause predicting never changes the learned parameters.
        virtual std::vector<double>predict(const std::vector<std::vector<double>>& X) const=0;
    };
}

#endif