#include "sklearn_cpp.h"

int main(){

    // LINEAR REGRESSION TEST CODE - CONCRETE DATA
    std::cout << "\n CONCRETE STRENGTH \n";
    std::vector<std::vector<double>> X_concrete;
    std::vector<double> Y_concrete;

    bool load = sklearn_cpp::DataLoader::load_csv("concrete.csv", X_concrete, Y_concrete);
    if(!load){
        std::cerr << "Load Failed\n";
        return 1;
    }
    std::cout << "Loaded: " << X_concrete.size() << " samples with: " << X_concrete[0].size() << " features.\n";

    sklearn_cpp::DataLoader::normalise(X_concrete);
    sklearn_cpp::LossMSE mse_loss;
    sklearn_cpp::linear_model::LinearRegression linear_model(X_concrete[0].size(), mse_loss, 0.001, 10000);
    linear_model.fit(X_concrete, Y_concrete);
    linear_model.printLoss(X_concrete, Y_concrete);

    // BINARY LOGISTIC TEST CODE - ECG DATA
    std::cout << "\n=====BINARY LOGISTIC REGRESSION - ECG Dataset=====\n";
    std::vector<std::vector<double>> X_diabetes;
    std::vector<double> Y_diabetes;

    load = sklearn_cpp::DataLoader::load_csv("ecg.csv", X_diabetes, Y_diabetes, -1, true);

    if(!load){
        std::cerr << "Load Failed\n";
        return 1;
    }
    std::cout << "Loaded: " << X_diabetes.size() << " samples with: " << X_diabetes[0].size() << " features.\n";

    sklearn_cpp::DataLoader::normalise(X_diabetes);
    sklearn_cpp::LossCrossEntropy ce_loss;
    sklearn_cpp::linear_model::LogRegBinary binary_model(X_diabetes[0].size(), ce_loss, 0.001, 10000);
    binary_model.fit(X_diabetes, Y_diabetes);
    binary_model.printLoss(X_diabetes, Y_diabetes);

    sklearn_cpp::LossCrossEntropy ce_loss;
    sklearn_cpp::linear_model::LogRegBinary model(X_diabetes[0].size(), ce_loss, 0.001, 10000);

    model.fit(X_diabetes, Y_diabetes);
    model.printLoss(X_diabetes, Y_diabetes);
    
    std::vector<double>predictions = model.predict(X_diabetes);
    std::cout << "SAMPLE PREDICTIONS\n";
    for(size_t i=0; i<10; i++){
        std::string result = predictions[i] == 1.0 ? "Diabetic" : "Not Diabetic";
        std::string actual = Y_diabetes[i] == 1.0 ? "Diabetic" : "Not Diabetic";
        std::cout << "Sample " << i << ": Predicted = " << result << " Actual = " << actual << "\n";
    }

    // SOFTMAX LOGISTIC TEST CODE - DIGIT DATA
    std::cout << "\n=====MULTICLASS LOGISTIC REGRESSION - Digits Dataset=====\n";
    std::vector<std::vector<double>> X_digits;
    std::vector<double> Y_digits;

    load = sklearn_cpp::DataLoader::load_csv("mnist_micro.csv", X_digits, Y_digits);
    if(!load){
        std::cerr << "Load Failed\n";
        return 1;
    }
    std::cout << "Loaded: " << X_digits.size() << " samples with: " << X_digits[0].size() << " features.\n";

    sklearn_cpp::DataLoader::normalise(X_digits);
    sklearn_cpp::linear_model::LogRegSoftmax softmax_model(X_digits[0].size(), 10, ce_loss, 0.001, 1000);
    softmax_model.fit(X_digits, Y_digits);
    softmax_model.printLoss(X_digits, Y_digits);

    predictions = softmax_model.predict(X_digits);
    std::cout << "\nSAMPLE PREDICTIONS\n";
    for(size_t i = 0; i < 10; i++){
        std::cout << "Sample " << i << ": Predicted digit = " << predictions[i] << "  Actual digit = " << Y_digits[i] << "\n";
    }

    return 0;
}