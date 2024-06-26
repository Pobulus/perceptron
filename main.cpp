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
size_t populationCap = 100;
NeuralNet allTimeBest({1,1,2});
double topPercentToKeep = 0.01;
double stagnationThreshold = 0.00001;

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
    auto nPopulationCap = configuration["populationCap"];
    if(nPopulationCap.readable()) nPopulationCap >> populationCap;

    auto nTopPercent = configuration["topPercentToKeep"];
    if(nTopPercent.readable()) nTopPercent >> topPercentToKeep;

    if(topPercentToKeep * populationCap < 1)
    {
        cerr << "No networks would be kept with this configuration!" << endl;
        return 1;
    }
    
    auto nStagnationThresh = configuration["stagnationThreshold"];
    if(nStagnationThresh.readable()) nStagnationThresh >> stagnationThreshold;

    auto nInConf = configuration["inputConf"];
    if(nInConf.readable()) nInConf >> inConf;

    auto nOutConf = configuration["outputConf"];
    if(nOutConf.readable()) nOutConf >> outConf;

    auto nDataSet = configuration["dataSetDir"];
    if(nDataSet.readable()) {
        string dataSetDirName;
        nDataSet >> dataSetDirName;
        trainingData = dataSetDirName;
    }
    
    cout << "\rPopulation cap: " << populationCap << endl;
    cout << "\rTop percent: " << topPercentToKeep << endl;
    cout << "\rStagnation threshold: " << stagnationThreshold << endl;
    cout << "\rTraining set: " << trainingData << endl;
    
     
    
    Loader loader( inConf, outConf);
    std::cout << "Loading tests..." << flush;
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
    vector<exercise> exercises;
    try {
        exercises = loader.loadExercises(trainingData);
    } catch(NNException ex) {
        cerr << ex.what() << endl;
        return 1;
    }
    cout << "Total exercise count: "<< exercises.size() << endl;

    vector<NeuralNet> population;
    auto nRestoreNetwork = configuration["restoreNetwork"];
    if(nRestoreNetwork.readable()) {
        string restoreFileName;
        nRestoreNetwork >> restoreFileName;
        NeuralNet restoredNet(networkShape);
        ifstream restoreFile(restoreFileName);
        if(restoreFile) {
            restoredNet.load(restoreFile);
            population.push_back(restoredNet);
            cout << "Restored " << restoreFileName  << endl;
        } else {
            cout << "Couldn't open file " << restoreFileName  << endl;
        }

    }

    long long int remainingSpace = populationCap-population.size();
    if(remainingSpace > 0) {
        cout << "Spawning new Neural nets" << flush;
        vector<NeuralNet> freshBlood = Breeder::get()->spawn(remainingSpace, networkShape);
        population.insert(population.end(), freshBlood.begin(), freshBlood.end());
    }
    cout << "\rInitial population ready." << endl;
    Teacher teacher;
    teacher.setExercises(exercises);
    int iters = 1;
    int generationCount = 0;
    double topAccuracy = 0.0;
    // return 0;
 do{
    while(iters-- > 0){
        teacher.setStudents(population);
        teacher.runTests();
        vector<pair<double, NeuralNet>> topStudents =  teacher.getTopStudents(topPercentToKeep);
        double accuracy = topStudents[0].second.getAccuracy();


        vector<NeuralNet> topNNs;


        for(auto student : topStudents){
            topNNs.push_back(student.second);
        }
        population = Breeder::get()->breed(topNNs);
        cout << "\rGen:"<< generationCount << " top accuracy: " << accuracy*100 << "%" << flush;
        // stagnation detection
        if( abs(topAccuracy - accuracy) < stagnationThreshold ) {
            cout << " -> stagnation! Mutating...";
            for(auto &net : population){
                net.mutate();
            }
        }
        if(accuracy > topAccuracy){
            topAccuracy = accuracy;
            allTimeBest = topNNs[0];
        }
        cout << endl;
        long long int remainingSpace = populationCap-population.size();
        if(remainingSpace > 0) {
            vector<NeuralNet> freshBlood = Breeder::get()->spawn(remainingSpace, networkShape);
            population.insert(population.end(), freshBlood.begin(), freshBlood.end());
        }
        generationCount++;
    }

        cout << "How many more generations? " <<  flush;
        cin >> iters;
}while(iters > 0);

    ofstream save("lastBest.net");
    allTimeBest.dump(save);
    std::cout << "saved" << std::endl;
    std::cout << "Would you like to test it out? " << flush;
    char resp = 0;
    cin >> resp;
    if(resp == 'y'){
        while(true){
        std::cout << "provide input" << std::endl;
        valarray<double> input(0.0, loader.inputsSize());
        for(int i = 0 ; i < loader.inputsSize(); i++){
            cin >> input[i];
        }
        valarray<double> output = allTimeBest.processInput(input);
        for(auto v : output){
            cout << v << "\t" ;
        }
        cout << endl;
        }
    }
    return 0;
}

