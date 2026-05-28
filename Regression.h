#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

class RegressionBase {

    protected:
        std::vector<std::vector<double>> x;
        std::vector<double> y;

        size_t m {};

        const double n {0.01};
        const int iteration {1000};
    
    public:
        virtual void fit(const std::vector<std::vector<double>>& X,
                        const std::vector<double>& Y) = 0;
        
        virtual ~RegressionBase() = default;
};

class Regression : public RegressionBase {

    protected:
        std::vector<double> w;
        double b {0.0};

        virtual void compute_gradients(std::vector<double>& derive_w, 
            double& derive_b) = 0;
        
        void solve() {
            for (size_t i = 0; i < iteration; i++) {
                std::vector<double> derive_w(m, 0.0);
                double derive_b {0.0};

                compute_gradients(derive_w, derive_b);

                for (size_t j = 0; j < m; j++) {
                    w[j] -= n * derive_w[j]; 
                };
                b  -= n * derive_b;
            };
        };
    
    public:         
        void fit(const std::vector<std::vector<double>>& X,
                const std::vector<double>& Y) override {
                
                x = X;
                y = Y;
                m = x[0].size();
                w.assign(m,0.0);

                solve();
                };
        
        void GradientResult() {
            std::cout << "===Gradient Descent Result===" << std::endl;
            for (size_t j = 0; j < w.size(); j++) {
                std::cout << "w" << j << " = " << w[j] << std::endl;
            };
            std::cout << "b = " << b << std::endl; 
        }

        virtual std::vector<double> predict(const std::vector<std::vector<double>>& X) = 0;
        
        virtual ~Regression() = default;
};



class MulticlassRegression : public RegressionBase {

    protected:
        size_t K;       // number of classes

        std::vector<std::vector<double>> w;
        std::vector<double> b;

        double lambda{0.01};

        virtual void compute_gradient(std::vector<std::vector<double>>& derive_w, 
            std::vector<double>& derive_b) = 0;
        
        void solve() {
            for (size_t i = 0; i < iteration; i++) {
                std::vector<std::vector<double>> derive_w{K, std::vector<double>(m, 0.0)};
                std::vector<double> derive_b(K, 0.0);

                compute_gradient(derive_w, derive_b);

                for (size_t j = 0; j < K; j++) {
                    for (size_t k = 0; k < m; k++) {
                        w[j][k] -= n * derive_w[j][k];
                    }; 
                    b[j]  -= n * derive_b[j];
                };
            };
        };
    public:
        MulticlassRegression(size_t K = 1) : K{K} {};
        void fit(const std::vector<std::vector<double>>& X,
                const std::vector<double>& Y) override{
                
                x = X;
                y = Y;
                m = x[0].size();
                w.assign(K, std::vector<double>(m,0.0));
                b.assign(K,0.0);

                solve();
                };

        virtual std::vector<double> predict(const std::vector<std::vector<double>>& X) = 0;
};

class Linear : public Regression {
    protected:
        void compute_gradients(std::vector<double>& derive_w, 
            double& derive_b) override;

    public:               
        virtual std::vector<double> predict(const std::vector<std::vector<double>>& X) override;
};

class Logistic : public Regression {
    protected:
        void compute_gradients(std::vector<double>& derive_w, 
            double& derive_b) override;
    public:   
        virtual std::vector<double> predict(const std::vector<std::vector<double>>& X) override;
};

class MulticlassLogistic : public MulticlassRegression {
    protected:
        void compute_gradient(std::vector<std::vector<double>>& derive_w, 
            std::vector<double>& derive_b) override;
    public:
        void GradientResult() {
            std::cout << "===Multiclass Gradient Descent Result===" << std::endl;
            for (size_t j = 0; j < K; j++) {
                std::cout << "Class " << j << ": " << std::endl;
                for (size_t k = 0; k < m; k++) {
                    std::cout << "w" << j << "," << k << " = " << w[j][k] << std::endl;
                }
                std::cout << "b" << j << " = " << b[j] << std::endl;
            }; 
        };
        virtual std::vector<double> predict(const std::vector<std::vector<double>>& X) override;
};