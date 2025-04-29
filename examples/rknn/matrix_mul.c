#define _CRT_SECURE_NO_DEPRECATE // Disables ridiculous "unsafe" warnigns on Windows
#include "ggml.h"
#include "ggml-cpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <inttypes.h>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

#define MAX_NARGS 2

// 矩阵乘法类型
#define MATMUL_FP16
// 调试，开启打印
#define MATMUL_DEBUG
// 每个维度的最大元素个数
#define MAX_ELEMENTS 32

float frand(void) {
    return (float)rand()/(float)RAND_MAX;
}

int irand(int n) {
    return rand()%n;
}

int irand_align32(int n) {
    if (n <= 32) return 32;  // 确保最小值是32
    
    int r = (rand() % (n - 32 + 1)) + 32;  // 生成32到n之间的随机数
    return r & (~0x1F);  // 确保结果是32的倍数
}

void get_random_dims(int64_t * dims, int ndims) {
    dims[0] = dims[1] = dims[2] = dims[3] = 1;

    for (int i = 0; i < ndims; i++) {
        dims[i] = 1 + irand(MAX_ELEMENTS);
    }
}

struct ggml_tensor * get_random_tensor(
        struct ggml_context * ctx0,
        int ndims,
        int64_t ne[],
        float fmin,
        float fmax) {
#ifdef MATMUL_FP16
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F16, ndims, ne);
#else
    struct ggml_tensor * result = ggml_new_tensor(ctx0, GGML_TYPE_F32, ndims, ne);
#endif

    switch (ndims) {
        case 1:
            for (int i0 = 0; i0 < ne[0]; i0++) {
            #ifdef MATMUL_FP16
                ((ggml_fp16_t *)result->data)[i0] = ggml_fp32_to_fp16(frand()*(fmax - fmin) + fmin);
            #else
                ((float *)result->data)[i0] = frand()*(fmax - fmin) + fmin;
            #endif
            }
            break;
        case 2:
            for (int i1 = 0; i1 < ne[1]; i1++) {
                for (int i0 = 0; i0 < ne[0]; i0++) {
                #ifdef MATMUL_FP16
                    ((ggml_fp16_t *)result->data)[i1*ne[0] + i0] = ggml_fp32_to_fp16(frand()*(fmax - fmin) + fmin);
                    // ((ggml_fp16_t *)result->data)[i1*ne[0] + i0] = ggml_fp32_to_fp16(0.01f*i0);
                #else
                    ((float *)result->data)[i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                #endif
                }
            }
            break;
        case 3:
            for (int i2 = 0; i2 < ne[2]; i2++) {
                for (int i1 = 0; i1 < ne[1]; i1++) {
                    for (int i0 = 0; i0 < ne[0]; i0++) {
                    #ifdef MATMUL_FP16
                        ((ggml_fp16_t *)result->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = ggml_fp32_to_fp16(frand()*(fmax - fmin) + fmin);
                    #else
                        ((float *)result->data)[i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                    #endif
                    }
                }
            }
            break;
        case 4:
            for (int i3 = 0; i3 < ne[3]; i3++) {
                for (int i2 = 0; i2 < ne[2]; i2++) {
                    for (int i1 = 0; i1 < ne[1]; i1++) {
                        for (int i0 = 0; i0 < ne[0]; i0++) {
                        #ifdef MATMUL_FP16
                            ((ggml_fp16_t *)result->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = ggml_fp32_to_fp16(frand()*(fmax - fmin) + fmin);
                        #else
                            ((float *)result->data)[i3*ne[2]*ne[1]*ne[0] + i2*ne[1]*ne[0] + i1*ne[0] + i0] = frand()*(fmax - fmin) + fmin;
                        #endif
                        }
                    }
                }
            }
            break;
        default:
            assert(false);
    };

    return result;
}

float get_element(const struct ggml_tensor * t, int idx) {
#ifdef MATMUL_FP16
    if (t->type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(((ggml_fp16_t *)t->data)[idx]);
    } else {
        return ((float *)t->data)[idx];
    }
#else
    return ((float *)t->data)[idx];
#endif
}

void set_element(struct ggml_tensor * t, int idx, float value) {
#ifdef MATMUL_FP16
    if (t->type == GGML_TYPE_F16) {
        ((ggml_fp16_t *)t->data)[idx] = ggml_fp32_to_fp16(value);
    } else {
        ((float *)t->data)[idx] = value;
    }
#else
    ((float *)t->data)[idx] = value;
#endif
}

// GGML不支持：ggml_compute_forward_out_prod_f16_f32
bool check_gradient(
        const char * op_name,
        struct ggml_context * ctx0,
        struct ggml_tensor * x[],
        struct ggml_tensor * f,
        int ndims,
        int nargs,
        float eps,
        float max_error_abs,
        float max_error_rel) {
    const int n_threads = 1;
    ggml_set_loss(f);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx0, GGML_DEFAULT_GRAPH_SIZE, true);
    ggml_build_forward_expand(gf, f);
    struct ggml_cgraph * gb = ggml_graph_dup(ctx0, gf);
    ggml_build_backward_expand(ctx0, ctx0, gb, false);

    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);
    ggml_graph_reset(gb);
    ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

    ggml_graph_dump_dot(gf, NULL, "test-grad0-forward.dot");
    ggml_graph_dump_dot(gb, gf,   "test-grad0-backward.dot");

    for (int i = 0; i < nargs; ++i) {
        const int64_t nelements = ggml_nelements(x[i]);
        for (int64_t k = 0; k < nelements; ++k) {
            // compute gradient using finite differences
            const float x0 = get_element(x[i], k);

            set_element(x[i], k, x0 + eps);
            ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

            const float f0 = ggml_get_f32_1d(f, 0);

            set_element(x[i], k, x0 - eps);
            ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

            const float f1 = ggml_get_f32_1d(f, 0);

            const float g0 = (f0 - f1)/(2.0f*eps);

            set_element(x[i], k, x0);

            // compute gradient using backward graph
            ggml_graph_reset(gb);
            ggml_graph_compute_with_ctx(ctx0, gb, n_threads);

            const float g1 = get_element(ggml_graph_get_grad(gb, x[i]), k);

            const float error_abs = fabsf(g0 - g1);
            const float error_rel = g0 != 0 ? fabsf(g0 - g1)/fabs(g0) : 0;

            if (error_abs > max_error_abs || error_rel > max_error_rel) {
                printf("%s: ndims=%d, i=%d, k=%" PRId64 ", g0=%f, g1=%f, error_abs=%f, error_rel=%f\n", op_name, ndims, i, k, g0, g1, error_abs, error_rel);
                assert(false);
            }
        }
    }

    return true;
}


float mat_get(const struct ggml_tensor * t, int i0, int i1, int i2, int i3) {
    const size_t nb0 = t->nb[0];
    const size_t nb1 = t->nb[1];
    const size_t nb2 = t->nb[2];
    const size_t nb3 = t->nb[3];
#ifdef MATMUL_FP16
    if (t->type == GGML_TYPE_F16) {
        return
            ggml_fp16_to_fp32(*((ggml_fp16_t*) ((char*)t->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3)));
    } else {
        return
            *((float*) ((char*)t->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3));
    }
#else
    return
        *((float*) ((char*)t->data + i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3));
#endif
}

bool check_mat_mul(
        const struct ggml_tensor * y,
        const struct ggml_tensor * x0,
        const struct ggml_tensor * x1) {
    const int64_t n00 = x0->ne[0];
    const int64_t n10 = x0->ne[1];
    const int64_t n20 = x0->ne[2];
    const int64_t n30 = x0->ne[3];

    const int64_t n01 = x1->ne[0];
    const int64_t n11 = x1->ne[1];
    const int64_t n21 = x1->ne[2];
    const int64_t n31 = x1->ne[3];

    const int64_t n02 = y->ne[0];
    const int64_t n12 = y->ne[1];
    const int64_t n22 = y->ne[2];
    const int64_t n32 = y->ne[3];
// #ifdef MATMUL_DEBUG
//     printf("x0: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", n00, n10, n20, n30);
//     for (int j = 0; j < n10; ++j) {
//         for (int i = 0; i < n00; ++i) {
//             printf("%6.3f ", mat_get(x0, i, j, 0, 0));
//         }
//         printf("\n");
//     }
//     printf("\n");

//     printf("x1: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", n01, n11, n21, n31);
//     for (int j = 0; j < n11; ++j) {
//         for (int i = 0; i < n01; ++i) {
//             printf("%6.3f ", mat_get(x1, i, j, 0, 0));
//         }
//         printf("\n");
//     }
//     printf("\n");

//     printf("y: [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n", n02, n12, n22, n32);
//     for (int j = 0; j < n12; ++j) {
//         for (int i = 0; i < n02; ++i) {
//             printf("%6.3f ", mat_get(y, i, j, 0, 0));
//         }
//         printf("\n");
//     }
// #endif
    for (int i3 = 0; i3 < n32; ++i3) {
        for (int i2 = 0; i2 < n22; ++i2) {
            for (int i1 = 0; i1 < n12; ++i1) {
                for (int i0 = 0; i0 < n02; ++i0) {
                    float sum = 0.0f;
                    for (int k = 0; k < n00; ++k) {
                        sum += mat_get(x0, k, i0, i2, i3) * mat_get(x1, k, i1, i2, i3);
                    }
                    if (fabsf(sum - mat_get(y, i0, i1, i2, i3)) > 1e-5) {
                        printf("error: i0=%d, i1=%d, i2=%d, i3=%d, sum=%f, y=%f\n",
                                i0, i1, i2, i3, sum, mat_get(y, i0, i1, i2, i3));
                        assert(false);
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

int main(int argc, const char ** argv) {
    struct ggml_init_params params = {
        .mem_size   = 128*1024*1024,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };

    int64_t ne_x1[4], ne_x0[4];

    // original loop: 500
    int niter = 500;
    const char *env = getenv("GGML_NLOOP");
    if (env != NULL) {
        niter = atoi(env);
    }
    if (argc > 1) {
        niter = atoi(argv[1]);
    }

    int n_threads = 1;

    const int64_t t_start = ggml_time_us();

    for (int iter = 0; iter < niter; ++iter) {
    #ifdef MATMUL_DEBUG
        printf("test-mul-mat0: iter:%d/%d\n", iter, niter);
    #endif
        struct ggml_context * ctx0 = ggml_init(params);

        struct ggml_tensor * x[MAX_NARGS];
    #ifdef MATMUL_DEBUG
        printf("********************Start to test mul mat*******************************\n");
    #endif
        // mul_mat
        {
            const int nargs = 1;

            for (int ndims = 2; ndims <= 4; ++ndims) {
                // 对于RKNN的CMN = AMK x BKN: 
                //  1. K和N需要为32倍数，即ne_x1[0]、ne_x0[0]和ne_x0[1]需要为32的倍数
                //  2. M可以为任意值，即ne_x1[1]可以任意
                //  3. 对于3D/4D的情况，ne_x1[2]和ne_x1[3]可以任意
                get_random_dims(ne_x1, 4);
                ne_x1[0] = irand_align32(MAX_ELEMENTS);

                ne_x0[1] = irand_align32(MAX_ELEMENTS);
                ne_x0[0] = ne_x1[0];    // 满足矩阵可乘（相等）
                ne_x0[2] = ne_x1[2];    // 满足矩阵可乘（可广播）
                ne_x0[3] = ne_x1[3];    // 满足矩阵可乘（可广播）

                x[0] = get_random_tensor(ctx0, ndims, ne_x0, -1.0f, 1.0f);
                x[1] = get_random_tensor(ctx0, ndims, ne_x1, -1.0f, 1.0f);

                ggml_set_param(ctx0, x[0]);

                struct ggml_tensor * m = ggml_mul_mat(ctx0, x[1], x[0]);
                struct ggml_tensor * f = ggml_sum(ctx0, m);
            #ifdef MATMUL_DEBUG
                printf("testing: mul_mat, [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] = [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] * [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
                           m->ne[0],    m->ne[1],    m->ne[2],    m->ne[3],
                        x[1]->ne[0], x[1]->ne[1], x[1]->ne[2], x[1]->ne[3],
                        x[0]->ne[0], x[0]->ne[1], x[0]->ne[2], x[0]->ne[3]);
            #endif
                assert(m->ne[0] == x[1]->ne[1]);
                assert(m->ne[1] == x[0]->ne[1]);
                assert(m->ne[2] == x[0]->ne[2]);
                assert(m->ne[3] == x[0]->ne[3]);
            #ifdef MATMUL_FP16 // 由于GGML暂未支持ggml_compute_forward_out_prod_f16_f32，所以这里不用校验梯度的方式  
                    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
                    ggml_build_forward_expand(gf, m);
                    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);
            #else
                if (ndims <= 2) {
                    check_gradient("mul_mat", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY);
                } else {
                    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
                    ggml_build_forward_expand(gf, m);
                    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);
                }
            #endif

                check_mat_mul(m, x[1], x[0]);
            }
        }
        // break;
    #ifdef MATMUL_DEBUG
        printf("********************Start to test mul mat transposed********************\n");
    #endif
        // mul_mat (transposed)
        {
            const int nargs = 1;

            for (int ndims = 2; ndims <= 4; ++ndims) {
                get_random_dims(ne_x1, 4);
                ne_x1[1] = irand_align32(MAX_ELEMENTS);

                ne_x0[1] = irand_align32(MAX_ELEMENTS);
                ne_x0[0] = ne_x1[1];    // 满足矩阵可乘（相等）
                ne_x0[2] = ne_x1[2];    // 满足矩阵可乘（可广播）
                ne_x0[3] = ne_x1[3];    // 满足矩阵可乘（可广播）

                x[0] = get_random_tensor(ctx0, ndims, ne_x0, -1.0f, 1.0f);
                x[1] = ggml_cont(ctx0, ggml_transpose(ctx0, get_random_tensor(ctx0, ndims, ne_x1, -1.0f, 1.0f)));

                ggml_set_param(ctx0, x[0]);

                struct ggml_tensor * m = ggml_mul_mat(ctx0, x[1], x[0]);
                struct ggml_tensor * f = ggml_sum(ctx0, m);
            #ifdef MATMUL_DEBUG
                printf("testing: mul_mat, [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] = [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] * [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]\n",
                           m->ne[0],    m->ne[1],    m->ne[2],    m->ne[3],
                        x[1]->ne[0], x[1]->ne[1], x[1]->ne[2], x[1]->ne[3],
                        x[0]->ne[0], x[0]->ne[1], x[0]->ne[2], x[0]->ne[3]);
            #endif
                assert(m->ne[0] == x[1]->ne[1]);
                assert(m->ne[1] == x[0]->ne[1]);
                assert(m->ne[2] == x[0]->ne[2]);
                assert(m->ne[3] == x[0]->ne[3]);
            #ifdef MATMUL_FP16  // 由于GGML暂未支持ggml_compute_forward_out_prod_f16_f32，所以这里不用校验梯度的方式
                    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
                    ggml_build_forward_expand(gf, m);
                    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);
            #else
                if (ndims <= 2) {
                    check_gradient("mul_mat", ctx0, x, f, ndims, nargs, 1e-3f, 1e-3f, INFINITY);
                } else {
                    struct ggml_cgraph * gf = ggml_new_graph(ctx0);
                    ggml_build_forward_expand(gf, m);
                    ggml_graph_compute_with_ctx(ctx0, gf, n_threads);
                }
            #endif
                check_mat_mul(m, x[1], x[0]);
            }
        }
        ggml_free(ctx0);
    }

    const int64_t t_end = ggml_time_us();
    printf("%s: total time = %f ms, average time = %f ms\n", __func__, 
            (t_end - t_start) / 1000.0, (t_end - t_start) / 1000.0 / niter);

    return 0;
}
