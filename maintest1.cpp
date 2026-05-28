#include "sklearn_cpp.h"

int main(){
    std::cout << "\n CONCRETE STRENGTH \n";
    std::vector<std::vector<double>> X_concrete;
    std::vector<double> Y_concrete;

    bool load = sklearn_cpp::DataLoader::load_csv("concrete.csv", X_concrete, Y_concrete, 8, true);

    if(!load){
        std::cerr << "Load Failed \n";
        return 1;
    }
    std::cout << "Loaded: " << X_concrete.size() << " samples with: " << X_concrete[0].size() << " features. \n";

    sklearn_cpp::DataLoader::normalise(X_concrete);

    sklearn_cpp::LossMSE mse_loss;
    sklearn_cpp::linear_model::LinearRegression linear_model(X_concrete[0].size(), mse_loss, 0.001, 10000);
    
    linear_model.fit(X_concrete, Y_concrete);
    linear_model.printLoss(X_concrete, Y_concrete);

}