#pragma once
#include <fstream>

#include "NeuralNet.hpp"
#include "Teacher.hpp"
#include <valarray>
#include <filesystem>
inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}


class Loader {
    vector<string> outputNames;
    vector<string> inputNames;
    filesystem::path cacheDir = "./cache/";
public:
    Loader( string inputsConfig, string outputsConfig){
        cout << "\rinput Config: " << inputsConfig << endl;
        cout << "output Config: " << outputsConfig << endl;
        string input;
        ifstream inputFile(inputsConfig);

        while(getline(inputFile, input)){
            inputNames.push_back(input);
        }
        string output;
        ifstream outputFile(outputsConfig);

        while(getline(outputFile, output)){
            outputNames.push_back(output);
        }
    }
    int inputsSize(){
        return inputNames.size();
    }
    int outputsSize(){
        return outputNames.size();
    }
    exercise loadExercise(istream& file) {
        valarray<double> prompt(0.0, inputNames.size());
        valarray<double> expected(0.0, outputNames.size());
        string inputLine;
        while(getline(file, inputLine)){
            stringstream ss(inputLine);
            double inputValue; ss >> inputValue;
            string inputName; getline(ss, inputName);
            ltrim(inputName);

            long long int  index = find(inputNames.begin(), inputNames.end(), inputName) - inputNames.begin();
            if(index > -1){
                // std::cout <<"input:" << inputName << endl;
                prompt[index] = inputValue;
            }
        }
        if(prompt.sum() == 0) cout << "Warning this input is all 0!" << endl;
        return {prompt, expected };
    }
    
    vector<exercise> loadExercises(filesystem::path dataSetDir) {
        vector<exercise> out;
         if(filesystem::create_directory(cacheDir) ) {
            cout << "Created cache dir" << endl;
        } 
        int inputId = 0;
        //for each subdirectory
        for(const auto& dirEntry : filesystem::recursive_directory_iterator(dataSetDir)) {
            if (filesystem::is_directory(dirEntry)) {
                inputId++;
                string subdirName =  dirEntry.path().filename();
                cout << "expected output: "<< subdirName << ' ' << flush;

                valarray<double> expected(0.0, outputNames.size());
                long long int outputIndex = find(outputNames.begin(), outputNames.end(), subdirName) - outputNames.begin();
                if(outputIndex > -1) {
                    expected[outputIndex] = 1.0;
                } else {
                    throw NNException("Loading error: "+ subdirName +" is not found in inputsConfig" );
                }
                auto cacheFile = cacheDir / (subdirName + ".cache");
                auto csvFile = cacheDir / (subdirName + ".csv");
                if (filesystem::exists(cacheFile) ) { // cache file exists, load from it
                    cout << "\tfound in cache " << flush;
                    ifstream cacheIn(cacheFile);
                    if(cacheIn){
                        int l = 0;
                        string cachedLine;
                        while(getline(cacheIn, cachedLine)){
                            if(cachedLine != ""){
                                stringstream lineStream(cachedLine);
                                valarray<double> prompt(0.0, inputNames.size());
                                for(size_t i = 0; i < inputNames.size(); i++) {
                                    double inVal;
                                    if(lineStream >> inVal) prompt[i] = inVal;
                                    else throw NNException("Error loading cached file: unexpected end of input. Is the inputs size correct?");
                                }    
                                out.push_back({prompt, expected});                           
                                l++;
                            }
                        }
                        cout <<  "loaded "<< l << " inputs from cache" << endl;    
                        
                    }else{
                        cerr << "Error opening cacheFile " << cacheFile << endl;
                    }
                    
                } else {
                    cout <<  subdirName << " not in cache, it will be generated" << endl;
                   
                    ofstream cacheOut(cacheFile);
                    ofstream csvOut(csvFile);
                    if(!cacheOut){throw NNException("Error creating cache file! Is the cache directory correct?");}
                    for(const auto& filePath : filesystem::recursive_directory_iterator(dirEntry)) {
                        if(filesystem::is_regular_file(filePath)) {
                            valarray<double> prompt(0.0, inputNames.size());
                            cout << "Loading file: " << filePath.path() << endl;
                            ifstream file(filePath.path());
                            string inputLine;
                            while(getline(file, inputLine)){
                                stringstream ss(inputLine);
                                double inputValue; ss >> inputValue;
                                string inputName; getline(ss, inputName);
                                ltrim(inputName);

                                long long int  index = find(inputNames.begin(), inputNames.end(), inputName) - inputNames.begin();
                                if(index > -1){
                                    // std::cout <<"input:" << inputName << endl;
                                    prompt[index] = inputValue;
                                }
                            }
                            if(prompt.sum() == 0) cout << "Warning this input is all 0!" << endl;
                            csvOut << inputId;
                            for(auto v : prompt){ cacheOut << v << ' '; csvOut << ',' << v ;}
                            cacheOut << endl; csvOut << endl;

                            out.push_back({prompt, expected});
                        }
                    }
                }
            }
        }
        return out;
    }
    void printResult(valarray<double> result) {
        for(size_t i = 0;  i < outputNames.size(); i++){
            cout << outputNames[i] << ":\t" << result[i] << endl;
        }
    }
};
