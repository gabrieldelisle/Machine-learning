#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <algorithm>
#include <iostream>

template <class T>
class Matrix {
  static_assert(std::is_move_constructible<T>::value, "Must be move-constructible!");
  static_assert(std::is_move_assignable<T>::value, "Must be move-assignable!");
  typedef T* iterator;
public:
   // constructors
   Matrix () {
     m_vec = nullptr;
     m_rows = 0;
     m_cols = 0;
     m_capacity = 0;
   }

   explicit Matrix (size_t sq) {
     if(sq == 0) {
      m_vec = nullptr;
     } else {
       m_vec = new T[sq * sq]();
     }
     m_rows = sq;
     m_cols = sq;
     m_capacity = sq * sq;
   }

   Matrix(std::initializer_list<T> list) {
    size_t square = (size_t)std::sqrt(list.size());
    if(square* square != list.size()) {
      throw std::out_of_range("not perfect square");
    } 
    m_rows = square;
    m_cols = square;
    if(square == 0) {
      m_vec = nullptr;
    } else {
      m_vec = new T[square*square]();
      auto first = list.begin();
      int i = 0;
      while(first != list.end()){
        m_vec[i] = (*first);
        first++;
        i++;
      }
    }


   }


   Matrix(size_t rows, size_t cols) {
    m_rows = rows;
    m_cols = cols;
    m_vec = new T[cols * rows]();
    m_capacity = rows * cols;
   }

   Matrix(const Matrix &cpy) {
    m_rows = cpy.m_rows;
    m_cols = cpy.m_cols;
    //T* old = m_vec;
    m_vec = new T[m_rows * m_cols]();
    //delete [] old;
    for (size_t i = 0; i< rows()* cols(); i++) {
      m_vec[i] = cpy.m_vec[i];
    }
   }
   //move constructor
   Matrix (Matrix && from) {
    m_rows = from.m_rows;
    m_cols = from.m_cols;
    m_capacity = from.m_capacity;
    m_vec = from.m_vec;
    from.m_rows = 0;
    from.m_cols = 0;
    from.m_capacity = 0;
    from.m_vec = nullptr;
   }



   // operators
   Matrix & operator= (const Matrix & rhs) {
    m_rows = rhs.m_rows;
    m_cols = rhs.m_cols;
    if(m_rows * m_cols <= m_capacity) {
      for(size_t i = 0; i < m_rows * m_cols; i++) {
        m_vec[i] = rhs.m_vec[i];
      }
    } else {
      T* tmp_vec = new T[m_rows * m_cols]();
      T* to_delete = m_vec;
      for(size_t i =0; i <m_rows * m_cols; i++) {
        tmp_vec[i] = rhs.m_vec[i];
      }
      m_vec = tmp_vec;
      delete [] to_delete;  
    }
    return *this;
   }

   Matrix & operator=(Matrix && from) {
    m_rows = from.m_rows;
    m_cols = from.m_cols;
    m_capacity = from.m_capacity;
    T* del = m_vec;
    m_vec = from.m_vec;
    delete [] del;
    from.m_rows = 0;
    from.m_cols = 0;
    from.m_capacity = 0;
    from.m_vec = nullptr;
    return *this;
   }

   const T& operator()(size_t row, size_t col) const{
     if(row < m_rows && col < m_cols){
      return m_vec[row * m_cols + col];
     } else {
      throw std::out_of_range("Out of range");
     }
    
   }

   T& operator()(size_t row, size_t col) {
     if(row < m_rows && col < m_cols){
      return m_vec[row * m_cols +  col];
     } else {
      throw std::out_of_range("Out of range");
     }
   }

   Matrix operator* (const Matrix & rhs)  const{
     if((*this).m_cols != rhs.m_rows) {
       throw std::out_of_range("wrong dimensions");
     }
     Matrix<T> matris;
     matris.m_rows = m_rows;
     matris.m_cols = rhs.cols();
     matris.m_vec = new T[rows()*rhs.cols()]();


     for(size_t i = 0; i < rows(); i++){
       for(size_t j= 0; j < rhs.cols(); j++) {
         T sum = T();
         for(size_t k = 0; k < cols(); k++) {
           sum += (*this)(i,k) * rhs(k, j);
         }
         matris(i,j) = sum;
       }
     }
     return matris;
   }

   Matrix  operator*(T scalar)  const{
    Matrix<T> matris = (*this);
    for(size_t i = 0; i< rows()*cols(); i++) {
      matris.m_vec[i] = m_vec[i] * scalar;
    }
    return matris;
   }

   Matrix  operator+(const Matrix & rhs) {
     if(!(cols() == rhs.cols() && rows() == rhs.rows())) {
      throw std::out_of_range("wrong dimensions");
     }
     Matrix<T> matris;
     T * tmp = new T[cols()*rows()]();
     matris.m_vec = tmp;
     matris.m_cols = cols();
     matris.m_rows = rows();

     for(size_t i = 0; i< cols()*rows(); i++) {
       tmp[i] = rhs.m_vec[i] + m_vec[i];
     }
     return matris;
   }

   Matrix operator-(const Matrix & rhs) {
     if(!(cols() == rhs.cols() && rows() == rhs.rows())) {
      throw std::out_of_range("wrong dimensions");
     }
     Matrix<T> matris;
     T * tmp = new T[cols()*rows()]();
     matris.m_vec = tmp;
     matris.m_cols = cols();
     matris.m_rows = rows();

     for(size_t i = 0; i< cols()*rows(); i++) {
       tmp[i] = m_vec[i] - rhs.m_vec[i];
     }
     return matris;
   }

   void operator*=(const Matrix & rhs) {
    (*this) = (*this) * rhs;
   }

   void operator*=(const T & scalar) {
    (*this) = (*this) * scalar;
   }

   void operator+=(const Matrix & rhs) {
    (*this) = (*this) + rhs;
   }

   void operator-=(const Matrix & rhs) { 
    (*this) = (*this) - rhs;
   }


   ~Matrix(){
    delete [] m_vec;
   }


   // methods
    size_t rows() const {
    return m_rows;
   }

   size_t cols() const {
    return m_cols;
   }

   void reset() {
    for(size_t i = 0; i < rows(); i++) {
      for(size_t j = 0; j <cols(); j++) {
        (*this)(j, j) = T();
      }
    }
   }

   iterator begin() {
    return m_vec;
   }

   iterator end() {
    return m_vec + rows()*cols();
   }

   void transpose() {
     T* new_vec = new T[rows()*cols()]();
     Matrix<T> temp;
     temp.m_vec = new_vec;
     temp.m_cols = rows();
     temp.m_rows = cols();
     temp.m_capacity = rows()* cols();



     for(size_t i = 0; i< rows(); i++ ) {
       for(size_t j = 0; j< cols(); j++) {
         temp(j,i)= (*this)(i,j);
       }
     }
     (*this) = temp;
   }

   void insert_row(size_t row){
     if(!(row < rows())) {
       throw std::out_of_range("out of range");
     }

     T * temp = new T[(rows()+1)* cols()]();
     iterator it = begin();
     for(size_t i = 0; i<=rows(); i++) {
         if(i == row) {
              continue;
        }
        for(size_t j = 0; j < cols(); j++) {
            temp[i*cols() + j] =(*it);
            it++;
        }
     }
     m_rows = rows()+1;
     T * del = m_vec;
     m_vec = temp;
     delete [] del;



   }

   void append_row(size_t row) {
     if(!(row < rows())) {
       throw std::out_of_range("out of range");
     }
     T * temp = new T[(rows()+1) * cols()]();
     iterator it = begin();
     for(size_t i = 0; i<=rows(); i++) {
         if(i == row+1) {
              continue;
        }
        for(size_t j = 0; j < cols(); j++) {
            temp[i*cols() + j] =(*it);
            it++;
        }
     }
     T* del = m_vec;
     m_vec = temp;
     delete[] del;
     m_rows +=1;
   }

   void remove_row(size_t row) {
     if(!(row < rows())) {
       throw std::out_of_range("out of range");
     }
     T * new_vec = new T[(rows()-1) * cols()]();
     std::copy(m_vec,m_vec+(row)*cols(), new_vec);
     std::copy(m_vec +(row+1)*cols(), end(), new_vec+row* cols());
     T* del = m_vec;
     m_vec = new_vec;
     delete[] del;
     m_rows -=1;

   }

   void insert_column(size_t col) {
     if(!(col < cols())) {
       throw std::out_of_range("out of range");
     }

     transpose();
     insert_row(col);
     transpose();
   }

   void append_column(size_t col) {
     transpose();
     append_row(col);
     transpose();
   }

   void remove_column(size_t col) {
     transpose();
     remove_row(col);
     transpose();
   }

   


private:
   size_t m_rows;
   size_t m_cols;
   size_t m_capacity;
   T * m_vec;

};

template <class T >
Matrix<T>  operator*(const T & scalar, const Matrix<T> & rhs) {
  return rhs * scalar;
}

template <class T>
Matrix<T> identity(unsigned int sq) {
  Matrix <T> matris(sq);
  for(size_t i = 0; i < sq; i++){
    matris(i,i) = 1;
  }
  return matris;
}
template <class T>
std::ostream & operator<< (std::ostream& os, const Matrix<T> &m){
  os << m.rows()<<" " ;
  os << m.cols()<<" " ;
  for(size_t i = 0; i< m.rows(); i++) {
    for(size_t j = 0; j <m.cols(); j++) {
      os << m(i,j) << " ";
    }
  }
  os << std::endl;

  return os;
}

template <class T>
std::istream & operator>> (std::istream& is, Matrix<T> & m){

  size_t rows, cols;
  is >> rows;
  is >> cols;
  Matrix<T> temp(rows,cols);
  for(size_t i =0; i< rows; i++) {
    for(size_t j = 0; j < cols; j++) {
      T val;
      is >> val;
      temp(i,j) = val;
    }
    m = temp;
  }
  return is;
}





