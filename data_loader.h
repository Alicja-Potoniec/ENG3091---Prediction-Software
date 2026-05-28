#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

namespace sklearn_cpp{

    class DataLoader{

        private:
        static std::vector<double> parse_row(const std::string& line){
            std::vector<double> row;
            std::stringstream ss(line);
            std::string cell;
            while(std::getline(ss, cell, ',')){
                row.push_back(std::stod(cell));
            }
            return row;
        }
        //SAFETY CHECK - Validating if the data has been loaded successfully.
        static bool validate (const std::vector<std::vector<double>>& X, const std::vector<double>& Y){
            if(X.empty() || Y.empty()){
                std::cerr << "Error: failed to load data \n";
                return false;
            }
            return true;
        }

        public:
        static bool load_csv(const std::string& filepath, std::vector<std::vector<double>>& X, std::vector <double>& Y, int target_column, bool has_header = true){

            std::ifstream file(filepath);
            if(!file.is_open()){
                std::cerr << "Error: Failed to open file." << filepath << "\n";
                return false;
            }
            std::string line;
            if(has_header) std::getline(file, line);

            while(std::getline(file, line)){
                if(line.empty()) continue;

                std::vector<double>features;
                for(size_t i=0; i<row.size(); i++){
                    if((int)i != target) features.push_back(row[i]);
                }
                X.push_back(features);
            }
            return validate (X,Y);
        }
        // NORMALISATION scales the feature to stabilise loss
        static void normalise(std::vector<std::vector<double>>& X){
            size_t n=X[0].size();
            for(size_t j=0; j<n; j++){
                double mean{0.0};
                for(size_t i=0; i<X.size(); i++) mean += X[i][j];
                mean /= X.size();

                double std_dev{0.0};
                for (size_t i=0; i<X.size(); i++)
                std_dev += (X[i][j] - mean) * (X[i][j] - mean);
                std_dev = std::sqrt(std_dev/ X.size());

                if(std_dev > 1e-8){
                    for(size_t i=0; i<X.size(); i++)
                    X[i][j] = (X[i][j] - mean)/ std_dev;
                }
            }
        }
    };
}

#endif