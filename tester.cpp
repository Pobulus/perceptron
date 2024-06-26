#include <iostream>
#include <fstream>
#include <random>
#include "NeuralNet.hpp"
#include "Breeder.hpp"
#include "Teacher.hpp"
#include "Loader.hpp"
#include <ryml_std.hpp>
#include <ryml.hpp>
#include <chrono>
#include <ctime>
//  configuration kept here for now
string inConf = "inputs.conf";
string outConf = "outputs.conf";
filesystem::path trainingData = "./training/";

std::string get_file_contents(const char *filename)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in) {
        std::cerr << "could not open " << filename << std::endl;
        return "";
    }
    std::ostringstream contents;
    contents << in.rdbuf();
    return contents.str();
}


int main ( int argc, char **argv ) {
    auto startTime = chrono::system_clock::now();
    time_t startTime_t = chrono::system_clock::to_time_t(startTime);
    cout << "Start: " << ctime(&startTime_t) << flush;
    std::cout << "Configuration..." << flush;
    std::string contents = get_file_contents("config.yaml");
    ryml::Tree configuration = ryml::parse_in_place(ryml::to_substr(contents));
       
    auto nInConf = configuration["inputConf"];
    if(nInConf.readable()) nInConf >> inConf;

    auto nOutConf = configuration["outputConf"];
    if(nOutConf.readable()) nOutConf >> outConf;

     
    
    Loader loader( inConf, outConf);
    std::cout << "Loading test..." << flush;

    vector<int> networkShape;
    auto nNetworkShape = configuration["networkShape"];
    if(nNetworkShape.readable()) {
        networkShape.push_back(loader.inputsSize());
        for(auto l : nNetworkShape.children()){
            int layerSize;
            l >> layerSize;
            networkShape.push_back(layerSize);
        }
        networkShape.push_back(loader.outputsSize());
        
    } else {
        networkShape = {loader.inputsSize(), 100, 100,  loader.outputsSize()};
    }
    cout << "\rnetwork shape: ";
    for(auto l : networkShape) {
        std::cout << l << " ";
    }
    cout << endl;
    exercise test;
    
    try {
        test = loader.loadExercise(cin);
    } catch(NNException ex) {
        cerr << ex.what() << endl;
        return 1;
    }
    NeuralNet network(networkShape);
    auto nRestoreNetwork = configuration["restoreNetwork"];
    if(nRestoreNetwork.readable()) {
        string restoreFileName;
        nRestoreNetwork >> restoreFileName;
        NeuralNet restoredNet(networkShape);
        ifstream restoreFile(restoreFileName);
        if(restoreFile) {
            restoredNet.load(restoreFile);
            network = restoredNet;
            cout << "Restored " << restoreFileName  << endl;
        } else {
            cout << "Couldn't open file " << restoreFileName  << endl;
        }
    }

    auto result = network.processInput(test.prompt);
    loader.printResult(result);
}

