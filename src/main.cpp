#include <iostream>
#include <string>
#include "../include/utils.h"
#include "../include/matrix.h"

using std::cout;
using std::endl;

void (*dp_func)(float *, float *, float *, int);

long long unsigned dot_product_1(float *ma, float *mb, float *mc, unsigned n) {
    start();
    regularMatrixMult(ma, mb, mc, n);
    long long unsigned sequential_time = stop();

    bool resultAB = validateAB(ma, mb, mc, n);
    fillMatrixC(mc, n);
    regularMatrixMult(mb, ma, mc, n);
    bool resultBA = validateBA(ma, mb, mc, n);
    if(!(resultAB && resultBA)) {
        sequential_time = 0;
    }
    return sequential_time;

}

long long unsigned dot_product_1_tr(float *ma, float *mb, float *mc, unsigned n) {
    start();
    regularMatrixMultTr(ma, mb, mc, n);
    long long unsigned sequential_time = stop();

    bool resultAB = validateAB(ma, mb, mc, n);
    fillMatrixC(mc, n);
    regularMatrixMultTr(mb, ma, mc, n);
    bool resultBA = validateBA(ma, mb, mc, n);
    if(!(resultAB && resultBA)) {
        sequential_time = 0;
    }
    return sequential_time;

}

long long unsigned dot_product_2(float *ma, float *mb, float *mc, unsigned n) {
    start();
    matrixMultIndexOrder1(ma, mb, mc, n);
    long long unsigned sequential_time = stop();

    bool resultAB = validateAB(ma, mb, mc, n);
    fillMatrixC(mc, n);
    matrixMultIndexOrder1(mb, ma, mc, n);
    bool resultBA = validateBA(ma, mb, mc, n);
    if(!(resultAB && resultBA)) {
        sequential_time = 0;
    }
    return sequential_time;
}

long long unsigned dot_product_3(float *ma, float *mb, float *mc, unsigned n) {
    start();
    matrixMultIndexOrder2(ma, mb, mc, n);
    long long unsigned sequential_time = stop();

    bool resultAB = validateAB(ma, mb, mc, n);
    fillMatrixC(mc, n);
    matrixMultIndexOrder2(mb, ma, mc, n);
    bool resultBA = validateBA(ma, mb, mc, n);
    if(!(resultAB && resultBA)) {
        sequential_time = 0;
    }
    return sequential_time;
}

long long unsigned dot_product_3_tr(float *ma, float *mb, float *mc, unsigned n) {
    start();
    matrixMultIndexOrder2Tr(ma, mb, mc, n);
    long long unsigned sequential_time = stop();

    bool resultAB = validateAB(ma, mb, mc, n);
    fillMatrixC(mc, n);
    matrixMultIndexOrder2Tr(mb, ma, mc, n);
    bool resultBA = validateBA(ma, mb, mc, n);
    if(!(resultAB && resultBA)) {
        sequential_time = 0;
    }
    return sequential_time;

}

int main(int argc, char *argv[]) {
    unsigned size = std::stoul(argv[1]);
    float ma[size*size], mb[size*size], mc[size*size];

    fillMatrixA(ma, size);
    fillMatrixB(mb, size);
    fillMatrixC(mc, size);
    
    #ifdef DOT_PR_1

    dp_func = &dot_product_1
    
    long long unsigned sequential_time = dot_product_1(ma, mb, mc, size);
    if(sequential_time == 0) {
        cout << "Regular Implementation Failed with wrong result!" << endl;
        return -1;
    }
    cout << "Sequential time: " << sequential_time << " usecs" << endl;

    #elif DOT_PR_1_TR

    long long unsigned sequential_time = dot_product_1_tr(ma, mb, mc, size);
    if(sequential_time == 0) {
        cout << "Regular Implementation Failed with wrong result!" << endl;
        return -1;
    }
    cout << "Sequential time: " << sequential_time << " usecs" << endl;

    #elif DOT_PR_2

    long long unsigned sequential_time = dot_product_2(ma, mb, mc, size);
    if(sequential_time == 0) {
        cout << "Regular Implementation Failed with wrong result!" << endl;
        return -1;
    }
    cout << "Sequential time: " << sequential_time << " usecs" << endl;
    
    #elif DOT_PR_3
    
    long long unsigned sequential_time = dot_product_3(ma, mb, mc, size);
    if(sequential_time == 0) {
        cout << "Regular Implementation Failed with wrong result!" << endl;
        return -1;
    }
    cout << "Sequential time: " << sequential_time << " usecs" << endl;
	
    #elif DOT_PR_3_TR

    long long unsigned sequential_time = dot_product_3_tr(ma, mb, mc, size);
    if(sequential_time == 0) {
        cout << "Regular Implementation Failed with wrong result!" << endl;
        return -1;
    }
    cout << "Sequential time: " << sequential_time << " usecs" << endl;

    #else
    cout << "No implementation Selected!" << endl;   
    return -1;
    #endif

    return 0;
}