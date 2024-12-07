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

// Split string
vector<string> split(const string& str, char delimiter) {
    vector<string> tokens;
    string token;
    istringstream tokenStream(str);
    while (getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Calculate cosine similarity
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
    if (argc < 2) {
        cerr << "Please provide the patient ID as a command line argument." << endl;
        return 1;
    }

    int patientIndex;
    try {
        patientIndex = stoi(argv[1]) - 1;  // Adjust patient ID to match 0-based index
    } catch (const invalid_argument& e) {
        cerr << "Invalid patient ID. Please enter a valid number." << endl;
        return 1;
    }

    // Read CSV file
    ifstream file("/home/baiyu323/Desktop/JiaqiZhang/Similarity/1234.csv");
    if (!file.is_open()) {
        cerr << "Unable to open file!" << endl;
        return 1;
    }

    string line;
    vector<vector<double>> data;
    vector<string> patientNames;

    // Read file data
    while (getline(file, line)) {
        vector<string> row = split(line, ',');
        if (row.empty()) continue;

        patientNames.push_back(row[0]);  // Assume the first column is the patient's name
        row.erase(row.begin());  // Remove the first column (patient name) to keep only features

        vector<double> numericRow;
        for (const string& value : row) {
            try {
                numericRow.push_back(stod(value));  // Attempt to convert string to double
            } catch (const invalid_argument& e) {
                cerr << "Unable to convert string to number: " << value << endl;  // Output error message
                numericRow.push_back(0.0);  // Fill with default value
            } catch (const out_of_range& e) {
                cerr << "Number out of range: " << value << endl;
                numericRow.push_back(0.0);  // Fill with default value
            }
        }
        data.push_back(numericRow);
    }

    // Check if patient ID is within range
    if (patientIndex < 0 || patientIndex >= static_cast<int>(data.size())) {
        cerr << "Patient ID is out of range. Valid range is 1 to " << data.size() << "." << endl;
        return 1;
    }

    // Store similarities between specified patient and others
    vector<pair<double, int>> similarities;

    for (size_t i = 0; i < data.size(); ++i) {
        if (i != static_cast<size_t>(patientIndex)) {  // Exclude self
            double similarity = cosineSimilarity(data[patientIndex], data[i]);
            similarities.emplace_back(similarity, i);
        }
    }

    // Sort similarities from high to low
    sort(similarities.begin(), similarities.end(), [](const pair<double, int>& a, const pair<double, int>& b) {
        return a.first > b.first;
    });

    // Build JSON output
    json output;
    output["patient_id"] = patientNames[patientIndex];
    output["matches"] = json::array();

    // Output the top three matching patients
    for (int i = 0; i < 3 && i < static_cast<int>(similarities.size()); ++i) {
        int index = similarities[i].second;
        json match;
        match["rank"] = i + 1;
        match["patient_id"] = patientNames[index];
        match["similarity"] = similarities[i].first;

        // Convert data to string, rounding to three decimal places
        string data_str;
        for (const auto& feature : data[index]) {
            data_str += to_string(round(feature * 1000.0) / 1000.0) + " ";  
        }
        match["data"] = data_str;

        output["matches"].push_back(match);
    }

    // Output matches within similarity range if specified
    if (argc == 4) {
        double lowerBound, upperBound;
        try {
            lowerBound = stod(argv[2]);
            upperBound = stod(argv[3]);
        } catch (const invalid_argument& e) {
            cerr << "Invalid similarity range. Please enter valid numbers." << endl;
            return 1;
        }

        json rangeOutput;
        rangeOutput["patient_id"] = patientNames[patientIndex];
        rangeOutput["range_matches"] = json::array();

        for (const auto& similarity : similarities) {
            if (similarity.first >= lowerBound && similarity.first <= upperBound) {
                int index = similarity.second;
                json match;
                match["patient_id"] = patientNames[index];
                match["similarity"] = similarity.first;

                // Convert data to string, rounding to three decimal places
                string data_str;
                for (const auto& feature : data[index]) {
                    data_str += to_string(round(feature * 1000.0) / 1000.0) + " ";  
                }
                match["data"] = data_str;

                rangeOutput["range_matches"].push_back(match);
            }
        }

        cout << rangeOutput.dump(4) << endl;  // Output JSON with an indentation of 4
    } else {
        cout << output.dump(4) << endl;  // Output JSON with an indentation of 4
    }

    return 0;
}

