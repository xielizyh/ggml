#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "ggml.h"
#include "rknn/rknn_wrap.h"
#include "rknn/include/rknn_api.h"
#include "rknn/include/rknn_matmul_api.h"

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
    info.K = K;
    info.N = N;
    info.type = matmul_type;
    info.B_layout = B_layout;
    info.AC_layout = AC_layout;
    info.iommu_domain_id = 0;
    
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
    memcpy(A->virt_addr, A_Matrix, io_attr.A.size);
    memcpy(B->virt_addr, B_Matrix, io_attr.B.size);

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
    memcpy(C_Matrix, C->virt_addr, io_attr.C.size);
exit:
    /* 释放NPU内存 */
    rknn_destroy_mem(ctx, A);
    rknn_destroy_mem(ctx, B);
    rknn_destroy_mem(ctx, C);

    /* 释放上下文 */
    rknn_matmul_destroy(ctx);

    return ret;
}


int rknn_vec_dot_f16(ggml_fp16_t * x, ggml_fp16_t * y, float * s, int n)
{
    return rknn_matrix_mul_f16(x, y, s, 1, n, 1);
}
