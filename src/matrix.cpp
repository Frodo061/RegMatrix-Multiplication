#include "../include/matrix.h"

void fillMatrixA (float **ma, unsigned n) {
	for (unsigned i = 0; i < n; ++i) {
		for (unsigned j = 0; j < n; ++j) {
			ma[i][j] = ((float) rand()) / ((float) RAND_MAX);
		}
	}
}

void fillMatrixB (float **mb, unsigned n) {
	for (unsigned i = 0; i < n; ++i) {
		for (unsigned j = 0; j < n; ++j) {
			mb[i][j] = 1;
		}
	}
}

void fillMatrixC(float **mc, unsigned n){
    for (unsigned i = 0; i < n; ++i) {
		for (unsigned j = 0; j < n; ++j) {
			mc[i][j] = 0;
		}
	}
}

void mcopy(float **m, float **c, unsigned n) {
    for (unsigned i = 0; i < n; ++i) {
		for (unsigned j = 0; j < n; ++j) {
			c[i][j] = m[i][j];
		}
	}
}

void transpose (float **m, unsigned n){
    float tmp;
    for (unsigned i = 0; i < n; i ++) {
        for (unsigned j = i+1; j < n; j ++) {
            tmp = m[i][j];
            m[i][j] = m[j][i];
            m[j][i] = tmp;
        }
    }
}

//Index Order i-j-k
void regularMatrixMult (float **ma, float **mb, float **mc, unsigned n) {
    
	for (unsigned i = 0; i < n; ++i) {
		for (unsigned j = 0; j < n; ++j) {
			for (unsigned k = 0; k < n; ++k) {
                mc[i][j] += ma[i][k] * mb[k][j];
			}
		}
	}
}

//Index Order i-j-k Transposed
void regularMatrixMultTr(float **ma, float **mb, float **mc, unsigned n) {
    transpose(mb,n);
	for (unsigned i = 0; i < n; ++i) {
		for (unsigned j = 0; j < n; ++j) {
			for (unsigned k = 0; k < n; ++k) {
                mc[i][j] += ma[i][k] * mb[j][k];
			}
		}
	}
}

//Index Order i-j-k Transposed Blocks
void regularMatrixMultBl(float **ma, float **mb, float **mc, unsigned n){
    transpose(mb, n);

    for( unsigned br = 0; br < n; br+=BLOCKSIZE)
        for( unsigned bc = 0; bc < n; bc+=BLOCKSIZE)
            for( unsigned i = 0; i < n; i++)
                for( unsigned j = br; j < br+BLOCKSIZE; j++) 
                    #pragma ivdep
                    for( unsigned k = bc; k < bc+BLOCKSIZE; k++)
                        mc[i][j] += ma[i][k] * mb[j][k];

}

//Index Order i-k-j
void matrixMultIndexOrder1(float **ma, float **mb, float **mc, unsigned n) {
    for(unsigned i = 0; i < n; ++i){
        for(unsigned k = 0; k < n; ++k){
            for(unsigned j = 0; j < n; ++j){
                mc[i][j] += ma[i][k] * mb[k][j];
            }
        }
    }
}

//Index Order i-k-j Blocks
void matrixMultIndexOrder1Bl(float **ma, float **mb, float **mc, unsigned n) {
    for(unsigned ii = 0; ii < n; ii += BLOCKSIZE) {
        for(unsigned jj = 0 ; jj < n; jj += BLOCKSIZE) {
            for(unsigned i = 0; i < n; i++) {
                for(unsigned k = ii; k < ii + BLOCKSIZE; k++){
                    for(unsigned j = jj; j < jj + BLOCKSIZE; j++){
                        mc[i][j] += ma[i][k] * mb[k][j];
                    }
                }
            }
        }
    }
}

//Index Order j-k-i
void matrixMultIndexOrder2(float **ma, float **mb, float **mc, unsigned n) {
    for(unsigned j = 0; j < n; ++j){
        for(unsigned k = 0; k < n; ++k){
            for(unsigned i = 0; i < n; ++i){
                mc[i][j] += ma[i][k] * mb[k][j];
            }
        }
    }
}


//Index Order j-k-i Transposed
void matrixMultIndexOrder2Tr(float **ma, float **mb, float **mc, unsigned n) {
    transpose(ma, n);
    transpose(mb, n);

    for(unsigned j = 0; j < n; ++j){
        for(unsigned k = 0; k < n; ++k){
            for(unsigned i = 0; i < n; ++i){
                mc[j][i] += ma[k][i] * mb[j][k];
            }
        }
    }
    transpose(mc, n);
}

//Index Order j-k-i Transposed Blocks
void matrixMultIndexOrder2Bl(float **ma,float **mb,float **mc, unsigned n) { 
    transpose(ma,n);
    transpose(mb,n);
    
    for( unsigned br = 0; br < n; br+=BLOCKSIZE)
        for( unsigned bc = 0; bc < n; bc+=BLOCKSIZE)
            for( unsigned j = 0; j < n; j++)
                for( unsigned k = br; k < br+BLOCKSIZE; k++) 
                    for( unsigned i = bc; i < bc+BLOCKSIZE; i++)
                        mc[j][i] += ma[k][i] * mb[j][k];

    transpose(mc, n);
}


bool validate(float **ma, float **mb, float **mc, unsigned n) {
    bool result = true;
    //all resulting columns should have the same values
    for(unsigned i = 0; i < n && result; i++) {
        float tmp = mc[i][0];
        for(unsigned j = 0; j < n && result; j++) {
            if(mc[i][j] != tmp) result = false;
        }
    }
    return result;
}