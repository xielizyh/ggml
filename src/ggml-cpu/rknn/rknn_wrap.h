#ifndef _RKNN_WRAP_H
#define _RKNN_WRAP_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

int rknn_matrix_mul_f16(ggml_fp16_t * A_Matrix, ggml_fp16_t * B_Matrix, float * C_Matrix, int M, int K, int N);

#ifdef __cplusplus
} // extern "C"
#endif

#endif //_RKNN_WRAP_H