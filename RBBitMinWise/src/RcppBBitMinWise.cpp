// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(openmp)]]
#include <fstream>
#include <sstream>
#include <chrono>
#include <math.h>
#include <vector>
#include <stdlib.h>
#include <random>
#include <Rcpp.h>

using namespace std;

typedef vector<int> Row;
typedef vector<Row> Matrix;

ostream& operator<<(ostream& os, const Matrix& m) {
    for (int i = 0; i < m.size(); i++) {
        for (int j = 0 ; j < m[i].size(); j++) {
            os << m[i][j] << " ";
        }
        os << endl;
    }
}

vector<string> explode(const string &text, char sep) {
    vector<string> tokens;
    size_t start = 0, end = 0;
    while ((end = text.find(sep, start)) != string::npos) {
        tokens.push_back(text.substr(start, end - start));
        start = end + 1;
    }
    tokens.push_back(text.substr(start));
    return tokens;
}

Matrix getPerms(int L, int k, int p) {
    Matrix A(L, Row(k));
    default_random_engine e;
    uniform_int_distribution<> die(0, p-1);
    for (int i = 0; i < L; i++) {
        for (int j = 0; j < k; j++) {
            A[i][j] = die(e);
        }
    }
    return A;
}

Matrix readPerms(const string permFileName) {
    ifstream fin;
    fin.open(permFileName);
    Matrix A;
    vector<string> splitString;
    for (string line; getline(fin, line);) {
        Row r;
        splitString = explode(line, ' ');
        for (int i=0; i < splitString.size(); i++) {
            r.push_back((int)strtol(splitString[i].c_str(), NULL, 10));
        }
        A.push_back(r);
    }
}


/***
 * @Brief Calculates the value of a universal hashing function
 *
 * @param t value to hash
 * @param D hashed values in range 0 to D-1
 * @param p A prime >> D
 * @param a A row vector defining the hash function polynomial
 */

int UkHash(int t, int D, int p, Row a) {
    int total = 0;
    for (int i = 0; i < a.size(); i++) {
        total += (a[i] * (t ^ i));
        total %= p;
    }
    total %= D;
    return total;
}


/**
 * @brief Compresses observation ready for prediction
 *
 * @param b The number of bits
 * @param L The number of blocks
 * @param p A prime >> L * b
 * @param line string containing svm format of new vector, there should be no classification
 * @param permFileName File containing the permutations used to compress data set
 *
 * @return Row vector containing positions of 1s in prediction vector
 */
// [[Rcpp::export]]
Row vectorProcess(string line, const int L, const int p, const int b, const string permFileName) {
    Row result;
    Matrix A = readPerms(permFileName);
    const int B = (int)pow(2.0,b);
    vector<int> j;
    vector<string> splitLine;
    vector<string> jx;
    splitLine = explode(line, ' ');
    for (int ctr = 0; ctr < splitLine.size(); ctr++) {
        jx = explode(splitLine[ctr], ',');
        j.push_back((int) strtol(jx[0].c_str(), NULL, 10));
    }
    int minHash = 1 << 30;
    int tempHash;
#pragma omp parallel for
    for (int l = 0; l < L; l++) {
        for (int ctr = 0; ctr < j.size(); ctr++) {
            tempHash = UkHash(j[ctr], B, p, (A)[l]);
            if (tempHash < minHash) {
                minHash = tempHash;
            }
        }
        result.push_back(l*B+(minHash % B)) ;
    }
#pragma omp taskwait
    return result;
}


void lineProcess(string line, string & result, const int L, const int p, const int b, Matrix * A) {
    // string stream result
    ostringstream convert;
    const int B = (int)pow(2.0,b);
    vector<int> j;
    //vector<int> x;
    vector<string> splitLine;
    vector<string> jx;
    splitLine = explode(line, ' ');
    convert << splitLine[0] << " ";
    for (int ctr = 1; ctr < splitLine.size(); ctr++) {
        jx = explode(splitLine[ctr], ',');
        j.push_back((int) strtol(jx[0].c_str(), NULL, 10));
        //x.push_back((int) strtol(jx[1].c_str(), NULL, 10));
    }
    int minHash = 1 << 30;
    int tempHash;
    //int minPos = 0;

    for (int l = 0; l < L; l++) {
        for (int ctr = 0; ctr < j.size(); ctr++) {
            tempHash = UkHash(j[ctr], B, p, (*A)[l]);
            if (tempHash < minHash) {
                minHash = tempHash;
                // minPos = j[ctr];
            }
        }
        convert << (l*B+(minHash % B)) << ":1 ";
    }
    result = convert.str();
    return;
}

/**
 * @brief Implementation of b bit minwise hashing on discrete 0 1 data
 *
 * @param b The number of bits
 * @param L The number of blocks
 * @param p A prime >> L * b
 * @param k Specifies the family of hashes to use, 4 recommended
 * @param inFileName Input file in svm format
 * @param permFileName File to output permutations, will be overwritten if already exists
 * @param outFileName Stores result in svm format, will be overwritten if alreay exists
 *
 * @return Creates files: hashes, containing the hash functions used; class, containing the class of each row; compress, which is bbit result;
 */
// [[Rcpp::export]]
void bBitCompress(const int b, const int L, const int p, const int k, const string inFileName, const string permFileName, const string outFileName) {
    Matrix A = getPerms(L, k, p);
    ofstream permsOut(permFileName);
    permsOut << A;
    permsOut.close();
    ifstream fin;
    fin.open(inFileName);
    ofstream resultOut(outFileName);
    string result;
#pragma omp parallel
    {
#pragma omp single
        {
            for (string line; getline(fin, line);) {
#pragma omp task private(result)
                {
                    lineProcess(line, result, L, p, b, &A);
                    resultOut << result << endl;
                    resultOut.flush();
                }
            }
        }
#pragma omp taskwait
    }
    fin.close();
    resultOut.close();
}