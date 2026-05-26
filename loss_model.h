#ifndef LOSS_MODEL_HPP
#define LOSS_MODEL_HPP

#include <vector>
# include <iostream>

namespace sklearn_cpp{

    class LossModel{
        public:
        virtual double computeLoss(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, const std::vector<double>& w, double b) = 0;

        virtual ~LossModel()=default;

    };
}

#endif