#ifndef LOSS_MODEL_HPP
#define LOSS_MODEL_HPP

//==========================================
// INTERFACE - loss_model.h
//==========================================
// Abstract interface that every loss function implements. 
// CONTRACT THAT LETS THE REGRESSION MODELS STAY IGNORANT OF WHICH LOSS NEEDS TO BE USED.
// They only need to hold the LossModel and can ask to computeLoss()
//
// DESIGN CHOICE
// Strategy pattern via an abstract base class, models inject the loss at construction phase. 
// Composition behaviour. Every subclass uses this interface. 

#include <vector>
#include <iostream>

namespace sklearn_cpp{

    // Abstract base class that nests all Loss functions
    // This class is not instantiated directly - it MUST be inherited from.
    class LossModel{
        
        // PURE VIRTUAL FUNCTION
        // Each subclass (LINEAR AND LOGISTIC) implement this. 
        // =0 makes the LossModel abstract. A bare LossModel cannot be created. 
        // Parameters are taken by const reference so no copies of large data matrices are made. 
        // the method is NON-CONST because the subclasses need to be allowed to update the internal data during computing. 
        public:
        virtual double computeLoss(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, const std::vector<double>& w, double b) = 0;

        // Destructor to enable safe cleanup
        // Critical for any polymorphic class to prevent memory bottlenecks.
        virtual ~LossModel()=default;

    };
}

#endif