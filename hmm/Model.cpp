#include "Model.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>


Model::Model(int nStates, int nObs)
{
	m_N = nStates;
	m_K = nObs;
	m_A = Matrix<double>(m_N, m_N);
	m_B = Matrix<double>(m_N, m_K);
	m_pi = Matrix<double>(1,m_N);

} 

void Model::add(int obs){
	m_obs.push_back(obs);
}


void Model::train(int loops, double random){

	std::srand(std::time(0));

	int T = m_obs.size();
   //Baum Welch algorithm: 
   //converges towards A, B and pi matrix to model the hidden markov model

   //initialize matrix A, B and pi
   for(int i=0; i<m_N; i++){
      double s = 0.0;
      for(int j=0; j<m_N; j++){
         m_A(i,j) = 1.0/m_N+(double(std::rand())/double(RAND_MAX)-0.5)/m_N*random;
         s+= m_A(i,j);
      }
      for(int j=0; j<m_N; j++){
         m_A(i,j)/=s;
      }
   }

   for(int i=0; i<m_N; i++){
      double s = 0.0;
      for(int j=0; j<m_K; j++){
         m_B(i,j) = 1.0/m_K+(double(std::rand())/double(RAND_MAX)-0.5)/m_K*random;
         s+= m_B(i,j);
      }
      for(int j=0; j<m_K; j++){
         m_B(i,j)/=s;
      }
   }

   double s = 0.0;
   for(int j=0; j<m_N; j++){
      m_pi(0,j) = 1.0/m_N+(double(std::rand())/double(RAND_MAX)-0.5)/m_N*random;;
      s+= m_pi(0,j);
   }
   for(int j=0; j<m_N; j++){
      m_pi(0,j)/=s;
   }

   //initialize coefficients
   double alpha[T][m_N];
   double scaling[T-1];
   double beta[T][m_N];
   double digamma[T-1][m_N][m_N];
   double gamma[T-1][m_N];

   double logProb=0;
   double oldLogProb= -1.0;


   for(int h=0; h<loops and logProb > oldLogProb; h++) {


      //alpha computation
      for(int i=0; i<m_N; i++) {
        alpha[0][i] = m_pi(0,i) * m_B(i,m_obs[0]);
      }
      scaling[0] = 0;
      for (int i = 0; i < m_N; ++i) {
         scaling[0] += alpha[0][i];
      }
      for (int i = 0; i < m_N; ++i) {
         alpha[0][i] /= scaling[0];
      }

      for(int t=1; t<T; t++) {
         for(int i=0; i<m_N; i++) {
            alpha[t][i] = 0 ;
            for(int j=0; j<m_N; j++) {
               alpha[t][i] += m_A(j,i) * alpha[t-1][j] ;
            }
            alpha[t][i] *= m_B(i,m_obs[t]) ;
            
         }

         scaling[t] = 0;
         for (int i = 0; i < m_N; ++i) {
            scaling[t] += alpha[t][i];
         }
         for (int i = 0; i < m_N; ++i) {
            alpha[t][i] /= scaling[t];
         }

      }

      
      
      // sum alpha
      double sumAlpha = 0.0 ;
      for(int i=0; i<m_N; i++) {
         sumAlpha += alpha[T-1][i] ;
      }

      //cout<<sumAlpha<<endl;

      // beta computation
      for(int i=0; i<m_N; i++) {
         beta[T-1][i] = 1/scaling[T-1];
      }

      for(int t=T-2; t>=0; t--) {
         for(int i=0; i<m_N; i++) {
            beta[t][i] = 0 ;
            for(int j=0; j<m_N; j++) {
               beta[t][i] += beta[t+1][j] * m_A(i,j) * m_B(j,m_obs[t+1]) ;
            }
            beta[t][i]/=scaling[t];
         }
      }

      //digamma computation
      for(int t=0; t<T-1; t++) {
         for(int i=0; i<m_N; i++) {
            for(int j=0; j<m_N; j++) {
               digamma[t][i][j] = alpha[t][i] * m_A(i,j) * m_B(j,m_obs[t+1]) * beta[t+1][j] / sumAlpha ;
            }
         }
      }


      //gamma computation
      for(int t=0; t<T-1; t++) {
         for(int i=0; i<m_N; i++) {
            gamma[t][i] = 0;
            for(int j=0; j<m_N; j++) {
               gamma[t][i] += digamma[t][i][j] ;
            }
         }
      }



      //new m_A matrix
      for(int i=0; i<m_N; i++) {
         for(int j=0; j<m_N; j++) {
            m_A(i,j) = 0;
            double d = 0;
            for(int t=0; t<T-1; t++) {
               m_A(i,j) += digamma[t][i][j] ;
               d += gamma[t][i];
            }
            m_A(i,j) /= d;
         }
      }


      //new m_B matrix
      for(int j=0; j<m_N; j++) {
         for(int k=0; k<m_K; k++) {
            m_B(j,k) = 0;
            double d = 0;
            for(int t=0; t<T-1; t++) {
               m_B(j,k) += (m_obs[t] == k) * gamma[t][j] ;
               d += gamma[t][j];
            }
            m_B(j,k) /= d;
         }
      }

      //new m_pi matrix
      for(int i=0; i<m_N; i++) {
         m_pi(0,i) = gamma[0][i];
      }

      //stop condition
      oldLogProb = logProb;
      logProb = 0;
      for(int t=0; t<T; t++) {
         logProb += scaling[t]; 
      }
   }



}

double Model::test(std::vector<int> obs){

	int T = obs.size();

	double alpha[T][m_N];
   double scaling[T-1];

   //alpha pass algorithm
   for(int i=0; i<m_N; i++) {
     alpha[0][i] = m_pi(0,i) * m_B(i,obs[0]);
   }
   scaling[0] = 0;
   for (int i = 0; i < m_N; ++i) {
      scaling[0] += alpha[0][i];
   }
   for (int i = 0; i < m_N; ++i) {
      alpha[0][i] /= scaling[0];
   }

   for(int t=1; t<T; t++) {
      for(int i=0; i<m_N; i++) {
         alpha[t][i] = 0 ;
         for(int j=0; j<m_N; j++) {
            alpha[t][i] += m_A(j,i) * alpha[t-1][j] ;
         }
         alpha[t][i] *= m_B(i,obs[t]) ;
         
      }

      scaling[t] = 0;
      for (int i = 0; i < m_N; ++i) {
         scaling[t] += alpha[t][i];
      }
      for (int i = 0; i < m_N; ++i) {
         alpha[t][i] /= scaling[t];
      }

   }

   double sum = 0.0 ;
   for(int i=0; i<m_N; i++) {
   	sum += alpha[T-1][i] ;
   }


   double sumlog = log(sum);
   for (int i = 0; i < T-1; ++i)
   {
   	sumlog+=log(scaling[i]);
   }

   return sumlog;
	
}

Matrix<double> Model::next(std::vector<int> obs){
   int T = obs.size();

   double alpha[T][m_N];
   double scaling[T-1];

   //alpha pass algorithm
   for(int i=0; i<m_N; i++) {
     alpha[0][i] = m_pi(0,i) * m_B(i,obs[0]);
   }
   scaling[0] = 0;
   for (int i = 0; i < m_N; ++i) {
      scaling[0] += alpha[0][i];
   }
   for (int i = 0; i < m_N; ++i) {
      alpha[0][i] /= scaling[0];
   }

   for(int t=1; t<T; t++) {
      for(int i=0; i<m_N; i++) {
         alpha[t][i] = 0 ;
         for(int j=0; j<m_N; j++) {
            alpha[t][i] += m_A(j,i) * alpha[t-1][j] ;
         }
         alpha[t][i] *= m_B(i,obs[t]) ;
         
      }

      scaling[t] = 0;
      for (int i = 0; i < m_N; ++i) {
         scaling[t] += alpha[t][i];
      }
      for (int i = 0; i < m_N; ++i) {
         alpha[t][i] /= scaling[t];
      }

   }

   Matrix<double> pi(m_pi);
   for(int i=0; i<m_N; i++){
      pi(0,i)=alpha[T-1][i];
   }

   pi*=m_A;
   Matrix<double> R(pi*m_B);

   double sum = 0.0 ;
   for(int i=0; i<m_K; i++) {
      sum += R(0,i) ;
   }
   R*=1/sum;

   return R;
}
