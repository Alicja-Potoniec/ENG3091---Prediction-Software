#ifndef BASEMODEL_HPP
#define BASEMODEL_HPP

#include <iostream>
#include <vector>

namespace sklearn_cpp{

    class BaseModel{

        protected:
        std::vector<double> w;
        double b{0.0};
        double learning_rate;
        int iterations;

        void gradient_descent(const std::vector<std::vector<double>>& X, const std::vector<double>& Y){
            size_t n{w.size()};
            for (int iter=0; iter<iterations; iter++){
                std::vector<double>grad_w(n, 0.0);
                double grad_b{0.0};
                compute_gradients(X, Y, grad_w, grad_b);
                for (size_t j=0; j<n; j++){
                    w[j] -= learning_rate*grad_w[j];
                }
                b -= learning_rate*grad_b;
            }
        }
        virtual void compute_gradients(const std::vector<std::vector<double>>& X, const std::vector<double>& Y, std::vector<double>& grad_w, double& grad_b) =0;
        
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
        public:
        BaseModel(int n_features, double lr=0.01, int iter=1000): w (n_features, 0.0), b(0.0), learning_rate(lr), iterations (iter){}

        virtual ~BaseModel() = default;

        void fit(const std::vector<std::vector<double>>& X, const std::vector<double>& Y){
            if (!validate(X, Y)) return;
            gradient_descent(X, Y);
        }
        virtual std::vector<double>predict(const std::vector<std::vector<double>>& X) const=0;
    };
}

#endif