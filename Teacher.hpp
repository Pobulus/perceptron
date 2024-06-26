#pragma once
#include <queue>
#include <vector>
#include "NeuralNet.hpp"
#include <algorithm>
#include <thread>

void print ( valarray<double> x ) {
    for ( double d : x ) {
        cout << d << ", ";
    }
}

struct exercise {
    valarray<double> prompt;
    valarray<double> expected;
};
double score ( valarray<double> result, valarray<double> expected ) {
    return ( 1- ( result-expected ).apply ( abs ) ).sum();
}

bool wasCorrect ( valarray<double>& result, valarray<double>& expected ) {
    size_t rInd = find(begin(result), end(result), result.max()) - begin(result);
    size_t eInd = find(begin(expected), end(result), 1) - begin(expected);
    return (rInd==eInd);
}


class Teacher {
    int threadCount = 4;
    vector<pair<double, NeuralNet>> students;
    vector<exercise> exercises;
    // void printProgress(size_t i) {
        // string prev = to_string(i);
        // std::cout  << string(prev.length(), '\b') <<  i+1  << std::flush;
    // }
    public:
    void setExercises ( vector<exercise> &ex ) {
        exercises = ex;
    }
    void setStudents ( vector<NeuralNet> newClass ) {

        students.clear();

        for ( auto &n : newClass ) {
            students.push_back ( {0.0, n} );
        }
    }
    void runTests() {
        if ( !exercises.size() ) {
            throw new NNException ( "Error: no exercises provided!" );
        }
        for(auto &n : students) {
            n.second.resetAccuracy();
        }
        for ( size_t i = 0; i < exercises.size(); i++) {
            std::cout << "\rRunning test "<< i <<"/" << exercises.size() << flush;
            // printProgress(i);
            test ( exercises[i] );
        }
        
    }
    void static testChunk(size_t index, int chunk,  exercise &Ex, vector<pair<double, NeuralNet>> &students) {
        for ( size_t i = index*chunk; i < (index+1)*chunk; i ++ ) {
            auto result = students[i].second.processInput ( Ex.prompt );
            bool correct = wasCorrect(result, Ex.expected);
            double sc = score ( result, Ex.expected );
            students[i].first += sc +  10 * ((int)correct+1);
            students[i].second.updateAccuracy(correct);
    }   }
    void test ( exercise &Ex ) {

        vector<thread> threads;
        size_t chunk = students.size() /threadCount;
        for(int t = 0; t < threadCount; t++){
            thread th ( testChunk, t, chunk, ref(Ex), ref(students) );
            threads.push_back(move(th));
        }
        for(auto &t : threads){
            t.join();
        }
    }
    vector<pair<double, NeuralNet>> getTopStudents ( double percentage ) {
        size_t count = ( size_t ) students.size() *percentage;
        if (count < 1) cerr << "Warning: this percentage will result in keeping just " << count << " nets!" << endl;  
        sort ( students.begin(), students.end(),  [] ( const auto& lhs, const auto& rhs ) {
            return lhs.first > rhs.first;
        } );
        return vector<pair<double, NeuralNet>> ( students.begin(), students.begin()+count );
    }
};
