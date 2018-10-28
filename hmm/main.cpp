#include "Model.hpp"
#include <vector>


int main() {
   //training observation
   std::vector<int> obs1 = {0,1,2,3,0,1,2,3,2,1,2,3};
   //test observation
   std::vector<int> obs2 = {0,1,2,3,2,1,2,3};

   //number of virtual states
   int N=6;
   //number of observation classes
   int K=4;

   // create model
   Model hmm(N,K);
   //add observation
   for (int i = 0; i < obs1.size(); ++i)
   {
   	hmm.add(obs1[i]);
   }
   hmm.train(50, 0.01);
   std::cout<<"logarithmic likelihood for observing obs2: "<<hmm.test(obs2)<<std::endl;
   Matrix<double> P(hmm.next(obs1));
   std::cout<<"Probabilities for next observation :"<<std::endl;
   for (int i = 0; i < K; ++i)
   {
      std::cout<<i<<": "<<P(0,i)<<std::endl;
   }

   return 0;

}
