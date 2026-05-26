#ifndef LOSS_MODEL_HPP
#define LOSS_MODEL_HPP

#include <vector>
#include <iostream>

namespace sklearn_cpp{

    // Abstract base class that nests all Loss functions
    // This class is not instantiated directly - it MUST be inherited from.
    class LossModel{
        
        // PURE VIRTUAL FUNCTION
        // Each subclass (LINEAR AND LOGISTIC) MUST implement this. 
        public:
        virtual double computeLoss(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, const std::vector<double>& w, double b) = 0;

        // Destructor to enable safe cleanup
        virtual ~LossModel()=default;

    };
}

#endif