#ifndef _MODEL_HPP_
#define _MODEL_HPP_

#include <vector>
#include "matrix.hpp"

class Model
{
/*
This class implements a Hiden Markov Model (HMM)

The model is a sequence of hidden states, each of these states producing an observation:

X0 -> X1 -> X2 -> X3 -> ...
 |     |     |     |
o0    o1    o2    o3

The model is composed of:
    - a vector pi representing the initial probabilities of being in each state
    - a transition matrix A : multiply the states vector by A to obtain the next states probabilities
    - a transition matrix B : multiply the states vector by B to obtain the probabilities of observations 
*/
private:
    //matrix of state transition
    Matrix<double> m_A;

    //matrix of observation transition
    Matrix<double> m_B;

    //vector of initial state
    Matrix<double> m_pi;

    //number of states
    int m_N;

    //number of observation classes
    int m_K;

    //observations
    std::vector<int> m_obs;

public: 

    //constructors
    Model()=default;
    Model(int nStates, int nObs);

    //add an observation to the model
    void add(int obs);

    //train the model computing matrices from stored observations
    void train(int loops, double random);

    //compute the probability of having a specific observation with this model
    double test(std::vector<int> obs);

    //compute the probability of each observations for next step, from stored observations
    Matrix<double> next(std::vector<int> obs);
};

#endif
