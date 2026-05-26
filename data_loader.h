#ifndef DATA_LOADER_HPP
#define DATA_LOADER_HPP

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

namespace sklearn_cpp{

    class DataLoader{

        public:
        static bool load_csv(const std::string& filepath, std::vector<std::vector<double>>& X, std::vector <double>& Y, int target_column, bool has_header = true);

        private:
        static std::vector<double> parse_row(const std::string& line);

        //SAFETY CHECK - Validating if the data has been loaded successfully.
        static bool validate (const std::vector<std::vector<double>>& X, const std::vector<double>& Y);
    };
}

#endif