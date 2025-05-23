#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "ggml.h"
#include "rknn/rknn_wrap.h"
#include "rknn/include/rknn_api.h"
#include "rknn/include/rknn_matmul_api.h"

static void matrix_print(const char* name, const float* matrix, int rows, int cols)
{
    printf("Matrix %s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%6.3f ", (double)matrix[i * cols + j]);
        }
        printf("\n");
    }
}

static void matrix_print_fp16(const char* name, const ggml_fp16_t* matrix, int rows, int cols) 
{
    printf("Matrix %s (%dx%d):\n", name, rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float val = ggml_fp16_to_fp32(matrix[i * cols + j]);
            printf("%6.3f ", (double)val);
        }
        printf("\n");
    }
}

static void matrix_transpose_fp16(const ggml_fp16_t* src, ggml_fp16_t* dst, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

static void matrix_pad_fp16(const ggml_fp16_t* src, ggml_fp16_t* dst, int rows, int cols, int pad_rows, int pad_cols)
{
    for (int i = 0; i < pad_rows; ++i) {
        for (int j = 0; j < pad_cols; ++j) {
            if (i < rows && j < cols) {
                dst[i * pad_cols + j] = src[i * cols + j];
            } else {
                dst[i * pad_cols + j] = ggml_fp32_to_fp16(0.0f);
            }
        }
    }
}

// 先填充再转置：pad_rows原始矩阵要填充的行数，pad_cols原始矩阵要填充的列数
static void matrix_pad_and_transpose_fp16(const ggml_fp16_t* src, ggml_fp16_t* dst, int rows, int cols, int pad_rows, int pad_cols)
{
    for (int i = 0; i < pad_rows; ++i) {
        for (int j = 0; j < pad_cols; ++j) {
            if (i < rows && j < cols) {
                dst[j * pad_rows + i] = src[i * cols + j];
            } else {
                dst[j * pad_rows + i] = ggml_fp32_to_fp16(0.0f);
            }
        }
    }
}

// 先转置再去填充：unpad_rows原始矩阵要去填充的行数，unpad_cols原始矩阵要去填充的列数
static void matrix_unpad_and_transpose(const float* src, float* dst, int rows, int cols, int unpad_rows, int unpad_cols)
{
    (void)(rows);

    for (int i = 0; i < unpad_rows; ++i) {
        for (int j = 0; j < unpad_cols; ++j) {
            dst[j * unpad_rows + i] = src[i * cols + j];
        }
    }
}

static int dim_align32(int dim)
{
    return (dim + 31) & (~31);
}

// GGML: C = A*B^T, C also is transposed
int rknn_matrix_mul_f16(ggml_fp16_t * A_Matrix, ggml_fp16_t * B_Matrix, float * C_Matrix, int M, int K, int N)
{
    int ret;
    int AC_layout, B_layout;
    rknn_matmul_type matmul_type;
    rknn_matmul_ctx ctx;
    rknn_matmul_info info;
    rknn_matmul_io_attr io_attr;
    rknn_tensor_mem *A, *B, *C;

    AC_layout = RKNN_MM_LAYOUT_NORM;
    B_layout = RKNN_MM_LAYOUT_NORM;
    matmul_type = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;   // fp16*fp16=fp32

    /* 初始化矩阵信息 */
    memset(&info, 0, sizeof(rknn_matmul_info));
    info.M = M;
    info.K = N; //!!! GGML矩阵乘法是需要将B矩阵转置后再进行乘法运算，所以这里的K和N需要交换
    info.N = K;
    info.type = matmul_type;
    info.B_layout = B_layout;
    info.AC_layout = AC_layout;
    info.iommu_domain_id = 0;
 #if 0   
    // RKNN矩阵乘法限制：1. fp16; 2. K < 2048; 3. K和N必须是32倍数，且大于等于32
    if (info.K % 32 || info.K > 2048 || info.N % 32) {
        fprintf(stderr, "RKNN: K=%d, N=%d is not supported!\n", info.K, info.N);
        return -1;
    }
#else   // 支持任意K和N
    if (info.K > 2048) {
        fprintf(stderr, "RKNN: K=%d(must be <=2048) is not supported!\n", info.K);
        return -1;
    }
    
    info.K = dim_align32(info.K);
    info.N = dim_align32(info.N);
#endif
    
    // printf("M=%d, K=%d, N=%d\n", M, K, N);
    // matrix_print_fp16("A", (ggml_fp16_t*)A_Matrix, M, K);
    // matrix_print_fp16("B", (ggml_fp16_t*)B_Matrix, K, N);

    /* 初始化矩阵乘法上下文 */
    memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));
    ret = rknn_matmul_create(&ctx, &info, &io_attr);
    if (ret < 0) {
        fprintf(stderr, "rknn_matmul_create fail! ret=%d\n", ret);
        return -1;        
    }

    /* 设置矩阵乘法运算的NPU核心 */
    ret = rknn_matmul_set_core_mask(ctx, RKNN_NPU_CORE_AUTO);
    if (ret < 0) {
        fprintf(stderr, "rknn_matmul_set_core_mask fail! ret=%d\n", ret);
        goto exit;
    }
    
    /* 分配矩阵NPU内存 */
    A = rknn_create_mem(ctx, io_attr.A.size);
    if (A == NULL) {
        fprintf(stderr, "rknn_create_mem fail!\n");
        goto exit;
    }
    B = rknn_create_mem(ctx, io_attr.B.size);
    if (B == NULL) {
        fprintf(stderr, "rknn_create_mem fail!\n");
        goto exit;
    }
    C = rknn_create_mem(ctx, io_attr.C.size);
    if (C == NULL) {
        fprintf(stderr, "rknn_create_mem fail!\n");
        goto exit;
    }

    /* 拷贝到NPU内存 */
    // memcpy(A->virt_addr, A_Matrix, io_attr.A.size);
    // transpose_fp16(B_Matrix, B->virt_addr, K, N);
    matrix_pad_fp16(A_Matrix, A->virt_addr, M, N, info.M, info.K);
    matrix_pad_and_transpose_fp16(B_Matrix, B->virt_addr, K, N, info.N, info.K);
    // matrix_print_fp16("A", (ggml_fp16_t*)A->virt_addr, info.M, info.K);
    // matrix_print_fp16("B^T", (ggml_fp16_t*)B->virt_addr, info.K, info.N);

    /* 设置矩输入/输出内存到矩阵乘法上下文 */
    ret = rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
    if (ret < 0) {
        fprintf(stderr, "rknn_matmul_set_io_mem fail! ret=%d\n", ret);
        goto exit;
    }
    ret = rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
    if (ret < 0) {
        fprintf(stderr, "rknn_matmul_set_io_mem fail! ret=%d\n", ret);
        goto exit;
    }
    ret = rknn_matmul_set_io_mem(ctx, C, &io_attr.C);
    if (ret < 0) {
        fprintf(stderr, "rknn_matmul_set_io_mem fail! ret=%d\n", ret);
        goto exit;
    }

    /* 执行矩阵相乘 */
    ret = rknn_matmul_run(ctx);
    if (ret < 0) {
        fprintf(stderr, "rknn_matmul_run fail! ret=%d\n", ret);
        goto exit;
    }

    /* 从NPU内存拷贝到CPU内存 */
    // transpose_fp32(C->virt_addr, C_Matrix, info.M, info.N);
    matrix_unpad_and_transpose(C->virt_addr, C_Matrix, info.M, info.N, M, K);
    // matrix_print("C", C_Matrix, M, K);
exit:
    /* 释放NPU内存 */
    rknn_destroy_mem(ctx, A);
    rknn_destroy_mem(ctx, B);
    rknn_destroy_mem(ctx, C);

    /* 释放上下文 */
    rknn_matmul_destroy(ctx);

    return ret;
}