#include "sklearn_cpp.h"

int main(){

    // ===================================================
    // TEST 1 - LINEAR REGRESSION - CONCRETE STRENGTH DATA
    // ===================================================
    // Tests the linearRegression model on the concrete compressive strength dataset.
    std::cout << "\n CONCRETE STRENGTH \n";
    std::vector<std::vector<double>> X_concrete;
    std::vector<double> Y_concrete;

    bool load = sklearn_cpp::DataLoader::load_csv("concrete.csv", X_concrete, Y_concrete);
    if(!load){
        std::cerr << "Load Failed\n";
        return 1;
    }
    std::cout << "Loaded: " << X_concrete.size() << " samples with: " << X_concrete[0].size() << " features.\n";
    
    // Normalise features to zero mean and unit variance before training. Required for gradient descent to converge
    // stably across features of different scales. 
    sklearn_cpp::DataLoader::normalise(X_concrete);

    // Construct the model with MSE loss, a learning rate of 0.001 and 10000 iterations.
    sklearn_cpp::LossMSE mse_loss;
    sklearn_cpp::linear_model::LinearRegression linear_model(X_concrete[0].size(), mse_loss, 0.001, 10000);
    linear_model.fit(X_concrete, Y_concrete);
    linear_model.printLoss(X_concrete, Y_concrete);

    // ==============================================
    // TEST 2 - BINARY LOGISTIC REGRESSION - ECG DATA
    // ==============================================
    // Tests the LogRegBinary model on ECG signal data.
    // Classified as Normal (1) or Abnormal (0). 
    std::cout << "\n=====BINARY LOGISTIC REGRESSION - ECG Dataset=====\n";
    std::vector<std::vector<double>> X_ecg;
    std::vector<double> Y_ecg;

    load = sklearn_cpp::DataLoader::load_csv("ecg.csv", X_ecg, Y_ecg);

    if(!load){
        std::cerr << "Load Failed\n";
        return 1;
    }
    std::cout << "Loaded: " << X_ecg.size() << " samples with: " << X_ecg[0].size() << " features.\n";

    // CLASS IMBALANCE CHECK - verifies the dataset is not heavily skewed toward one label.
    int count1 = 0;
    for(double label : Y_ecg) if(label == 1.0) count1++;
    std::cout << "Class 1 (Normal): " << count1 << " / " << Y_ecg.size() << "\n";
    std::cout << "Class 0 (Abnormal): " << Y_ecg.size() - count1 << " / " << Y_ecg.size() << "\n";

    sklearn_cpp::DataLoader::normalise(X_ecg);

    // Binary cross-entropy loss is used because it is the standard loss function for 
    // two class classification problems.
    sklearn_cpp::LossCrossEntropy ce_loss;
    sklearn_cpp::linear_model::LogRegBinary binary_model(X_ecg[0].size(), ce_loss, 0.001, 10000);
    binary_model.fit(X_ecg, Y_ecg);
    binary_model.printLoss(X_ecg, Y_ecg);
    
    // Sample predictions printed from index 2950 for a mix of both classes
    std::vector<double>predictions = binary_model.predict(X_ecg);
    std::cout << "SAMPLE PREDICTIONS\n";
    for(size_t i=2950; i<2965; i++){
        std::string result = predictions[i] == 1.0 ? "Normal" : "Abnormal";
        std::string actual = Y_ecg[i] == 1.0 ? "Normal" : "Abnormal";
        std::cout << "Sample " << i << ": Predicted = " << result << " Actual = " << actual << "\n";
    }

    // ==========================================================
    // TEST 3 - MULTICLASS LOGISTIC REGRESSION - MNIST DIGIT DATA
    // ==========================================================
    // Tests the LogRegSoftmax model on a subset of MNIST digit dataset.
    std::cout << "\n=====MULTICLASS LOGISTIC REGRESSION - Digits Dataset=====\n";
    std::vector<std::vector<double>> X_digits;
    std::vector<double> Y_digits;

    load = sklearn_cpp::DataLoader::load_csv("mnist_micro.csv", X_digits, Y_digits, -1, false);
    if(!load){
        std::cerr << "Load Failed\n";
        return 1;
    }
    std::cout << "Loaded: " << X_digits.size() << " samples with: " << X_digits[0].size() << " features.\n";

    sklearn_cpp::DataLoader::normalise(X_digits);

    // A separate loss instance is created for the softmax model to keep the binary and 
    // multiclass loss objects independent.
    // Learning rate is increased to 0.01.
    sklearn_cpp::LossCrossEntropy ce_loss_multi;
    sklearn_cpp::linear_model::LogRegSoftmax softmax_model(X_digits[0].size(), 10, ce_loss_multi, 0.01, 1000);
    softmax_model.fit(X_digits, Y_digits);
    softmax_model.printLoss(X_digits, Y_digits);

    // Sample predictions predicted
    predictions = softmax_model.predict(X_digits);
    std::cout << "\nSAMPLE PREDICTIONS\n";
    for(size_t i = 295; i < 310; i++){
        std::cout << "Sample " << i << ": Predicted digit = " << predictions[i] << "  Actual digit = " << Y_digits[i] << "\n";
    }

    return 0;
}