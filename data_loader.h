#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

// ================================================================================================
// DATA LOADER - data_loader.h
// Static utility class for loading and pre-processing .csv data files
//=================================================================================================
//
// This loader holds no state and all of the methods are static (does not get instatiated)
// Designed to group load_csv and normalise in the way that mirrors sklearn's pre processing API. 
// 
// The call is: DataLoader::load_csv

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

namespace sklearn_cpp{

    class DataLoader{
        
        // -------------------------
        // PARSE ROW - Static helper
        // -------------------------
        // Splits the CSV line on commas and converts each token to a double.
        // Returns full row as a vector<double>
        private:
        static std::vector<double> parse_row(const std::string& line){
            std::vector<double> row;
            std::stringstream ss(line);                                 
            std::string cell;
            while(std::getline(ss, cell, ',')){                         // wraps the line string so that std::getline can use ',' as a delimeter.
                row.push_back(std::stod(cell));                         // std::stod performs  the string-to-double conversion.It throws std::invalic_argument on non-numeric text.
                                                                        // A production loader wraps this in to try to identify or skip bad rows.
            }
            return row;
        }
        //SAFETY CHECK - Validating if the data has been loaded successfully.
        // -------------------------------------------------------------------
        // Called at the end of load_csv to confirm at least some rows were loaded successfully. An empty X & Y indicates that every row was skipped,
        // or that the file was empty
        static bool validate (const std::vector<std::vector<double>>& X, const std::vector<double>& Y){
            if(X.empty() || Y.empty()){
                std::cerr << "Error: failed to load data \n";
                return false;
            }
            return true;
        }

        public:
        //----------------------------
        // CSV LOADER
        //----------------------------
        // Reads the .CSV file, X & Y are passed by non-const reference. The function writes the results directly into the caller's vectors.
        //
        // Parameters:
        //-------------
        // filepath         - Path to the .CSV file
        // X                - Output feature matrix (rows = samples [n], cols = features [m])
        //                    Passed by reference and populated inside of this function.
        // Y                - output label vector. Passed by reference. 
        // target_column    - Specifies which column index is meant to be treated as a label.
        //                    Default -1 means the last column.
        // has_header       - If true, the first column is discarded. True by default.
        //                    Set as false for raw data where the first line includes data.
        //
        // Design note:
        //--------------
        // An alternative would be to return std::pair<X,Y> or a custom struct. Reference parameters were chosen to avoid copying potentially large matrices. 
        // A return-by-value approach would be equally efficient but instead went with the familiar approach. 
        // 
        // std::ifstream - closes the underlying file descriptor automatically after it goes out of scope. 
        //                 hence no explicit file.close() call is necessary. 

        static bool load_csv(const std::string& filepath, std::vector<std::vector<double>>& X, std::vector <double>& Y, int target_column = -1, bool has_header = true){


            std::ifstream file(filepath);
            if(!file.is_open()){
                std::cerr << "Error: Failed to open file." << filepath << "\n";
                return false;
            }
            std::string line;
            if(has_header) std::getline(file, line);        // skip the header row if the file has one.

            while(std::getline(file, line)){
                if(line.empty()) continue;                  // skip any blank lines. To preserve reusability with any .CSV files.

                std::vector<double>row = parse_row(line);
                if(row.empty()) continue;

                int target = (target_column == -1) ? (int)row.size() - 1 : target_column;       // Resolves the target column index for THIS row.
                                                                                                // - 1 means the LAST column. Lets the code to adapt automatically
                                                                                                // Only -1 is treated as sentinel
                Y.push_back(row[target]);

                // Copy every column except the target into the feature vector. 
                // "features" vector is built with non-target cells pushed into it. 
                std::vector<double>features;
                for(size_t i=0; i<row.size(); i++){
                    if((int)i != target) features.push_back(row[i]);
                }
                X.push_back(features);
            }
            return validate (X,Y);
        }
        //----------------------------------------------------
        // NORMALISATION scales the feature to stabilise loss
        //----------------------------------------------------
        // Rescales each FEATURE (column) to mean 0 and standard deviation 1
        // 
        // Gradient Descent converges much faster and stably when a feature shares common scale
        // Loss surface gets elongated when the data has varying magnitudes. 
        // With normalising a single learning rate works across all features.
        //---------------
        // ALTERNATIVE:
        // min-max scaling to [0, 1], approach taken by the sklearn library. 
        // Z-score was chosen because it handles different types of data uniformly. 
        static void normalise(std::vector<std::vector<double>>& X){
            size_t n=X[0].size();       // Number of features - columns
            for(size_t j=0; j<n; j++){
                double mean{0.0};
                for(size_t i=0; i<X.size(); i++) mean += X[i][j];
                mean /= X.size();

                // PASS 2 - Population Standard Deviation of column j
                double std_dev{0.0};
                for (size_t i=0; i<X.size(); i++)
                std_dev += (X[i][j] - mean) * (X[i][j] - mean);
                std_dev = std::sqrt(std_dev/ X.size());

                // GUARD: Only divide if there is variance in the column!
                // The 1e-8 threshold skips constant/ near-const columns. 
                // Prevents division by 0
                // Constant columns are left untouched (they carry no information for the classifier.)
                if(std_dev > 1e-8){
                    for(size_t i=0; i<X.size(); i++)
                    X[i][j] = (X[i][j] - mean)/ std_dev;
                }
            }
        }
    };
}

#endif