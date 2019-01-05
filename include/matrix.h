#include <iostream>

#define BLOCKSIZE 1

void fillMatrixA (float *ma, unsigned n);
void fillMatrixB (float *mb, unsigned n);
void fillMatrixC(float *mc, unsigned n);
void mcopy(float *m, float *c, unsigned n);
void regularMatrixMult (float *ma, float *mb, float *mc, unsigned n);
void regularMatrixMultTr (float *ma, float *mb, float *mc, unsigned n);
void regularMatrixMultBl(float *ma, float *mb, float *mc, unsigned n);
//Index Order i-k-j
void matrixMultIndexOrder1(float *ma, float *mb, float *mc, unsigned n);
void matrixMultIndexOrder1Bl(float *ma, float *mb, float *mc, unsigned n);
//Index Order j-k-i
void matrixMultIndexOrder2(float *ma, float *mb, float *mc, unsigned n);
void matrixMultIndexOrder2Tr(float *ma, float *mb, float *mc, unsigned n);
void matrixMultIndexOrder2Bl(float *ma,float *mb,float *mc, unsigned n);
bool validate(float *ma, float *mb, float *mc, unsigned n);