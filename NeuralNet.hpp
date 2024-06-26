#pragma once
#include <numeric>
#include <vector>
#include <valarray>
#include <iostream>
#include <sstream>
#include <random>
#include <cmath>
#include "nnException.hpp"
using namespace std;

const double BETA = 1.1254;
double activationFunctionSigmoid ( double n ) {
    return 1/ ( 1+exp ( -n*BETA ) );
};

std::random_device dev;
std::mt19937 rng ( dev() );

normal_distribution mutationDist {-0.25, 0.25};
double mutateWeight(double w){
    return w + mutationDist(rng);
}
uniform_real_distribution<double> zero ( 0,0 );

class NeuralNetLayer {

    friend class NeuralNet;

    vector<valarray<double>> weightMatrix;
    double ( *activation ) ( double ) = activationFunctionSigmoid;
public:
    NeuralNetLayer (const NeuralNetLayer  &original){
        weightMatrix = original.weightMatrix;
    }
    NeuralNetLayer ( size_t connectionsPerNeuron = 0, size_t neuronCount = 0 ) {
        for ( size_t n = 0; n < neuronCount; n++ ) {
            weightMatrix.push_back ( valarray<double> ( 0.0, connectionsPerNeuron ) );
        }
    }
    void randomize ( uniform_real_distribution<double> dist ) {
        for ( auto &neuron : weightMatrix ) {
            for ( auto &w : neuron ) {
                w = dist ( rng );
            }
        }
    }
    void mutate(){
        for ( auto &neuron : weightMatrix ) {
            neuron.apply(mutateWeight);
        }
    }
    void setWeights ( vector<vector<double>> &input ) {
        // TODO put safety checks on matching sizes
        for ( size_t i = 0; i < weightMatrix.size(); i++ ) {

            weightMatrix[i] = valarray<double> ( input[i].data(), input[i].size() );
        }

    }
    valarray<double> process ( valarray<double> &input ) {
        valarray<double> output ( weightMatrix.size() );
        for ( size_t i = 0; i < weightMatrix.size(); ++i ) {
            output[i] = ( weightMatrix[i]*input ).sum(); // calculate weighed sums
        }

        return output.apply ( activation );

    };
    void printSelf ( ostream &out ) {
        for ( auto neuron : weightMatrix ) {
            for ( auto weight : neuron ) {
                out << weight << ' ';
            }
            out << endl;
        }
    }
    size_t size() {
        return weightMatrix.size();
    }
};
class NeuralNet {
    vector<NeuralNetLayer> layers;
    vector<int> layerSizes;
    double correctAnswers = 0;
    size_t testsTaken = 0; 
public:
    void updateAccuracy(int result){
        testsTaken++; correctAnswers += result;
    }
    void resetAccuracy(){
        correctAnswers = 0;
        testsTaken = 0;
    }
    double getAccuracy(){
    cout << "ca:" << correctAnswers<< " tt:"<< testsTaken << endl;
        return correctAnswers/testsTaken;
    }
    NeuralNet(const NeuralNet &original) {
        layerSizes = original.layerSizes;
        correctAnswers = original.correctAnswers;
        testsTaken = original.testsTaken;
        for(auto &layer : original.layers){
            layers.push_back(NeuralNetLayer(layer));
        }
    }
    void mutate()
    {
        for(auto &layer : layers) {
            layer.mutate();
        }
    }
    NeuralNet ( vector<int> sizes ) : layerSizes ( sizes ) {
        if ( sizes.size() < 2 ) {
            throw NNException ( "Err Invalid NeuralNet size: 2 layers minimum! got" + to_string(sizes.size()) );
        }
        auto sizeIt = sizes.begin();
        int prev = *sizeIt++;
        while ( sizeIt != sizes.end() ) {
            if(*sizeIt <= 0) { 
                throw NNException("Err Invalid layer size 1 is minimum! got "+ to_string(*sizeIt));
            }
            layers.push_back ( NeuralNetLayer ( prev, *sizeIt ) );
            prev = *sizeIt++;
        }

    }
    valarray<double> processInput ( valarray<double> input ) {
        for ( auto &layer : layers ) {
            input = layer.process ( input );
        }
        return input;
    }
    void load ( istream &in ) {
        int i = 0;

        while ( in.peek() != EOF ) {
            string line = "";
            vector<vector<double>> data;

            do {

                getline ( in, line );

                if ( line == "" ) break;

                stringstream ss ( line );
                vector<double> dataLine;
                double tmp = 0;

                while ( ss >> tmp )
                    dataLine.push_back ( tmp );

                data.push_back ( dataLine );


            } while ( true );

            layers[i++].setWeights ( data );

        }


    }
    void randomize ( double min = 0, double max = 1 ) {
        for ( auto &layer : layers ) {
            layer.randomize ( uniform_real_distribution<double> ( min, max ) );
        }
    }
    void dump ( ostream &out ) {
        for ( auto &layer : layers ) {
            layer.printSelf ( out );
            out << endl;
        }
    }
//     bias is not limitted to 0-1 but should be used with caution
    NeuralNet breed ( NeuralNet &partner, double bias = 0.5 ) {
        if ( layerSizes != partner.layerSizes ) {
            throw NNException ( "Err Gene mismatch: layer sizes don't match" );
        }
        NeuralNet offspring = layerSizes;
        // set offspring weight to weighed averages between this and partner
        for ( size_t l = 0; l < layers.size();  l++ ) {
            for ( size_t i = 0; i < layers[l].size(); i++ ) {
                offspring.layers[l].weightMatrix[i]
                    = this->layers[l].weightMatrix[i] * ( 1-bias )
                      + partner.layers[l].weightMatrix[i] * bias;
            }
        }
        return offspring;
    }
};
