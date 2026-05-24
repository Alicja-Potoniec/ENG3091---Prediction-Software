#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>


class Regression {
    private:
        std::vector<std::vector<double>> x;
        std::vector<double> y;

        std::vector<double> w;
        double b {0.0};

        const double n {0.01};
        const int iteration {1000};

        size_t m;

    protected:
        virtual void compute_gradients(std::vector<double>& grad_w, 
            double& grad_b) = 0;
    public: 
        Regression(std::vector<std::vector<double>> x, std::vector<double> y, 
            std::vector<double> w) : x{x}, y{y}, w{w}, m{x.size()} {};
        
        void gradientDescent() {
            for (size_t i = 0; i < iteration; i++) {
                std::vector<double> derive_w(m, 0.0);
                double derive_b = 0.0;

                compute_gradients(derive_w, derive_b);

                for (size_t j = 0; j < m; j++) {
                    w[j] -= n * derive_w[j]; 
                }
                b  -= n * derive_b;
            }
        }
    public:
        virtual void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& Y) = 0;

        virtual std::vector<double> predict(const std::vector<std::vector<double>>& X) = 0;
        
        virtual ~Regression() = default;
};


class Logistic : public Regression {
    private:
    double sigmoid{0.0};
    public:
        void compute_gradients(std::vector<double>& grad_w, 
            double& grad_b) override;

        void fit(const std::vector<std::vector<double>>& X,
             const std::vector<double>& Y) override;
        
        virtual std::vector<double> predict(const std::vector<std::vector<double>>& X) override;
};