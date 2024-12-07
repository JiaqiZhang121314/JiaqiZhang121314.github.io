#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include "json.hpp"  // Ensure json.hpp is in the same directory

using namespace std;
using json = nlohmann::json;

vector<string> split(const string& str, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(str);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

double cosineSimilarity(const vector<double>& vec1, const vector<double>& vec2) {
    double dotProduct = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        dotProduct += vec1[i] * vec2[i];
        normA += vec1[i] * vec1[i];
        normB += vec2[i] * vec2[i];
    }
    return (normA == 0 || normB == 0) ? 0 : (dotProduct / (sqrt(normA) * sqrt(normB)));
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        cerr << "Please provide the patient ID and similarity range (lower and upper bounds) as command line arguments." << endl;
        return 1;
    }

    int patientIndex;
    double lowerBound, upperBound;

    try {
        patientIndex = stoi(argv[1]) - 1;  // Adjust patient ID to match 0-based index
        lowerBound = stod(argv[2]);
        upperBound = stod(argv[3]);
        if (lowerBound > upperBound) {
            cerr << "Invalid similarity range: lower bound should be less than upper bound." << endl;
            return 1;
        }
    } catch (const invalid_argument&) {
        cerr << "Invalid arguments. Please enter valid numbers." << endl;
        return 1;
    }

    ifstream file("/home/baiyu323/Desktop/JiaqiZhang/Similarity/1234.csv");
    if (!file.is_open()) {
        cerr << "Unable to open file!" << endl;
        return 1;
    }

    string line;
    vector<vector<double>> data;
    vector<string> patientNames;

    while (getline(file, line)) {
        vector<string> row = split(line, ',');
        if (row.empty()) continue;

        patientNames.push_back(row[0]);
        row.erase(row.begin());

        vector<double> numericRow;
        for (const string& value : row) {
            try {
                numericRow.push_back(stod(value));
            } catch (const invalid_argument&) {
                numericRow.push_back(0.0);
            } catch (const out_of_range&) {
                numericRow.push_back(0.0);
            }
        }
        data.push_back(numericRow);
    }

    if (patientIndex < 0 || patientIndex >= static_cast<int>(data.size())) {
        cerr << "Patient ID is out of range." << endl;
        return 1;
    }

    vector<pair<double, int>> similarities;
    for (size_t i = 0; i < data.size(); ++i) {
        if (i != static_cast<size_t>(patientIndex)) {
            double similarity = cosineSimilarity(data[patientIndex], data[i]);
            similarities.emplace_back(similarity, i);
        }
    }

    json output;
    output["patient_id"] = patientNames[patientIndex];
    output["matches"] = json::array();

    for (const auto& pair : similarities) {
        if (pair.first > lowerBound && pair.first < upperBound) {
            json match;
            match["patient_id"] = patientNames[pair.second];
            match["similarity"] = pair.first;

            string data_str;
            for (const auto& feature : data[pair.second]) {
                data_str += to_string(round(feature * 1000.0) / 1000.0) + " ";
            }
            match["data"] = data_str;

            output["matches"].push_back(match);
        }
    }

    cout << output.dump(4) << endl;

    return 0;
}

