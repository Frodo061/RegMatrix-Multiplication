#include <iostream>
#include <string>
#include <string.h>
#include "../include/utils.h"
#include "../include/matrix.h"

using std::cout;
using std::endl;

float *ma, *mb, *mc;

void (*dp_func)(float *, float *, float *, unsigned);

int main(int argc, char *argv[]) {

    if(argc < 3) {
        cout << "usage: bin/main size(32|128|1024|2048)" "type(time|l1mr|l2mr|l3mr|flops|vecops)" << endl;
        return -1;
    }

    unsigned size = std::stoul(argv[1]);
    const char *type=strdup(argv[2]);

    #if defined(DOT_PR_1_BL) || defined(DOT_PR_2_BL) || defined(DOT_PR_3_BL)
    if(argc == 4) {
        #undef BLOCKSIZE
        #define BLOCKSIZE atoi(argv[3])    
    } else {
        cout << "No size for blocks was provided! Using default block size of 1" << endl;
    }
    #endif
    
    #ifdef DOT_PR_1
    dp_func = &regularMatrixMult;
    #elif DOT_PR_1_TR
    dp_func = &regularMatrixMultTr;
    #elif DOT_PR_1_BL
    dp_func = &regularMatrixMultBl;
    #elif DOT_PR_2
    dp_func = &matrixMultIndexOrder1;
    #elif DOT_PR_2_BL
    dp_func = &matrixMultIndexOrder1Bl;
    #elif DOT_PR_3
    dp_func = &matrixMultIndexOrder2;
    #elif DOT_PR_3_TR
    dp_func = &matrixMultIndexOrder2Tr;
    #elif DOT_PR_3_BL
    dp_func = &matrixMultIndexOrder2Bl;
    #else
    cout << "No implementation Selected!" << endl;   
    return -1;
    #endif
    
    utils_setup_papi(8,type);
    for(unsigned i = 0; i < 8; i++){
        utils_init_matrices(&ma, &mb, &mc, size);
        clearCache();    
        utils_start_timer();
        utils_start_papi(type);
        dp_func(ma, mb, mc, size);
        utils_stop_papi(i, type);
        utils_stop_timer();
        if(!validate(ma, mb, mc, size)) {
            cout << "Matrix Multiplication failed: algorithm is incorrect" << endl;
            return -1;
        }
        utils_clean_matrices(&ma,&mb,&mc);
    }

    utils_results(type);

    return 0;
}