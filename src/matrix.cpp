#include "../include/matrix.h"

void fillMatrixA (float *ma, unsigned n) {
	for (unsigned i = 0; i < n; ++i) {
		for (unsigned j = 0; j < n; ++j) {
			ma[i * n + j] = ((float) rand()) / ((float) RAND_MAX);
		}
	}
}

void fillMatrixB (float *mb, unsigned n) {
	for (unsigned i = 0; i < n; ++i) {
		for (unsigned j = 0; j < n; ++j) {
			mb[i * n + j] = 1;
		}
	}
}

void fillMatrixC(float *mc, unsigned n){
    for (unsigned i = 0; i < n; ++i) {
		for (unsigned j = 0; j < n; ++j) {
			mc[i * n + j] = 0;
		}
	}
}

void mcopy(float *m, float *c, unsigned n) {
    for (unsigned i = 0; i < n; ++i) {
		for (unsigned j = 0; j < n; ++j) {
			c[i * n + j] = m[i * n + j];
		}
	}
}

void transpose (float *m, unsigned n){
    float tmp;
    for (unsigned i = 0; i < n; i ++) {
        for (unsigned j = i+1; j < n; j ++) {
            tmp = m[i*n + j];
            m[i*n + j] = m[j*n + i];
            m[j*n + i] = tmp;
        }
    }
}

//Index Order i-j-k
void regularMatrixMult (float *ma, float *mb, float *mc, unsigned n) {
    
	for (unsigned i = 0; i < n; i++) {
		for (unsigned j = 0; j < n; j++) {
			for (unsigned k = 0; k < n; k++) {
                mc[i * n + j] += ma[i * n + k] * mb[k * n + j];
			}
		}
	}
}

//Index Order i-j-k Transposed
void regularMatrixMultTr(float *ma, float *mb, float *mc, unsigned n) {
    transpose(mb,n);
	for (unsigned i = 0; i < n; i++) {
		for (unsigned j = 0; j < n; j++) {
			for (unsigned k = 0; k < n; k++) {
                mc[i * n + j] += ma[i * n + k] * mb[j * n + k];
			}
		}
	}
}

//Index Order i-j-k Transposed Blocks
void regularMatrixMultBl(float *ma, float *mb, float *mc, unsigned n){
    transpose(mb, n);

    for( unsigned br = 0; br < n; br+=BLOCKSIZE)
        for( unsigned bc = 0; bc < n; bc+=BLOCKSIZE)
            for( unsigned i = 0; i < n; i++)
                for( unsigned j = br; j < br+BLOCKSIZE; j++) 
                    #pragma ivdep
                    for( unsigned k = bc; k < bc+BLOCKSIZE; k++)
                        mc[i*n+j] += ma[i*n+k] * mb[j*n+k];

}

//Index Order i-k-j
void matrixMultIndexOrder1(float *ma, float *mb, float *mc, unsigned n) {
    for(unsigned i = 0; i < n; ++i){
        for(unsigned k = 0; k < n; ++k){
            for(unsigned j = 0; j < n; ++j){
                mc[i*n+j] += ma[i*n+k] * mb[k*n+j];
            }
        }
    }
}

void multiplicationIKJ (float** a, float** b, float** c, int size) {
	int i, k, j;
	for (i = 0; i < size; ++i)
		for (k = 0; k < size; ++k)
			for (j = 0; j < size; ++j)
				c[i][j] += a[i][k] * b[k][j];
}

//Index Order i-k-j Blocks
void matrixMultIndexOrder1Bl(float *ma, float *mb, float *mc, unsigned n) {
    for(unsigned ii = 0; ii < n; ii += BLOCKSIZE) {
        for(unsigned jj = 0 ; jj < n; jj += BLOCKSIZE) {
            for(unsigned i = 0; i < n; i++) {
                for(unsigned k = ii; k < ii + BLOCKSIZE; k++){
                    for(unsigned j = jj; j < jj + BLOCKSIZE; j++){
                        mc[i * n + j] += ma[i * n + k] * mb[k * n + j];
                    }
                }
            }
        }
    }
}

//Index Order j-k-i
void matrixMultIndexOrder2(float *ma, float *mb, float *mc, unsigned n) {
    for(unsigned j = 0; j < n; j++){
        for(unsigned k = 0; k < n; k++){
            for(unsigned i = 0; i < n; i++){
                mc[i*n+j] += ma[i*n+k] * mb[k*n+j];
            }
        }
    }
}


//Index Order j-k-i Transposed
void matrixMultIndexOrder2Tr(float *ma, float *mb, float *mc, unsigned n) {
    transpose(ma, n);
    transpose(mb, n);

    for(unsigned j = 0; j < n; ++j){
        for(unsigned k = 0; k < n; ++k){
            for(unsigned i = 0; i < n; ++i){
                mc[j*n+i] += ma[k*n+i] * mb[j*n+k];
            }
        }
    }
    transpose(mc, n);
}

//Index Order j-k-i Transposed Blocks
void matrixMultIndexOrder2Bl(float *ma,float *mb,float *mc, unsigned n) { 
    transpose(ma,n);
    transpose(mb,n);
    
    for( unsigned br = 0; br < n; br+=BLOCKSIZE)
        for( unsigned bc = 0; bc < n; bc+=BLOCKSIZE)
            for( unsigned j = 0; j < n; j++)
                for( unsigned k = br; k < br+BLOCKSIZE; k++) 
                    for( unsigned i = bc; i < bc+BLOCKSIZE; i++)
                        mc[j*n+i] += ma[k*n+i] * mb[j*n+k];

    transpose(mc, n);
}


bool validate(float *ma, float *mb, float *mc, unsigned n) {
    bool result = true;
    //all resulting columns should have the same values
    for(unsigned i = 0; i < n*n && result; i += n) {
        float tmp = mc[i];
        for(unsigned j = 0; j < n && result; j++) {
            if(mc[i + j] != tmp) result = false;
        }
    }
    return result;
}