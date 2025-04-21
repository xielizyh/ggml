#include "ggml.h"
#include "ggml-cpu.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;

    // the context to define the tensor information (dimensions, size, memory data)
    struct ggml_context * ctx;
};

// initialize the tensors of the model in this case two matrices 2x2
void load_model(simple_model & model, ggml_fp16_t * a, ggml_fp16_t * b, int rows_A, int cols_A, int rows_B, int cols_B) {
    size_t ctx_size = 0;
    {
        ctx_size += rows_A * cols_A * ggml_type_size(GGML_TYPE_F16); // tensor a
        ctx_size += rows_B * cols_B * ggml_type_size(GGML_TYPE_F16); // tensor b
        ctx_size += 2 * ggml_tensor_overhead(), // tensors
        ctx_size += ggml_graph_overhead(); // compute graph
        ctx_size += 1024; // some overhead
    }

    struct ggml_init_params params {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ false, // NOTE: this should be false when using the legacy API
    };

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F16, cols_A, rows_A);
    model.b = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F16, cols_B, rows_B);

    memcpy(model.a->data, a, ggml_nbytes(model.a));
    memcpy(model.b->data, b, ggml_nbytes(model.b));
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(const simple_model& model) {
    struct ggml_cgraph  * gf = ggml_new_graph(model.ctx);

    // result = a*b^T
    // Pay attention: ggml_mul_mat(A, B) ==> B will be transposed internally
    // the result is transposed
    struct ggml_tensor * result = ggml_mul_mat(model.ctx, model.a, model.b);

    ggml_build_forward_expand(gf, result);
    return gf;
}

// compute with backend
struct ggml_tensor * compute(const simple_model & model) {
    struct ggml_cgraph * gf = build_graph(model);

    int n_threads = 1; // number of threads to perform some operations with multi-threading

    ggml_graph_compute_with_ctx(model.ctx, gf, n_threads);

    // in this case, the output tensor is the last one in the graph
    return ggml_graph_node(gf, -1);
}

int main(void) {
    ggml_time_init();

    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 4, cols_A = 2;

    float matrix_A[rows_A * cols_A] = {
        0.2, 0.8,
        0.5, 0.1,
        0.4, 0.2,
        0.8, 0.6
    };
    ggml_fp16_t matrix_A_f16[rows_A * cols_A];

    const int rows_B = 3, cols_B = 2;
    /* Transpose([
        1.0, 0.9, 0.5,
        0.5, 0.9, 0.4
    ]) 2 rows, 3 cols */
    float matrix_B[rows_B * cols_B] = {
        1.0, 0.5,
        0.9, 0.9,
        0.5, 0.4
    };
    ggml_fp16_t matrix_B_f16[rows_B * cols_B];

    // convert float to fp16
    for (int i = 0; i < rows_A * cols_A; i++) {
        matrix_A_f16[i] = ggml_fp32_to_fp16(matrix_A[i]);
    }
    for (int i = 0; i < rows_B * cols_B; i++) {
        matrix_B_f16[i] = ggml_fp32_to_fp16(matrix_B[i]);
    }
    
    simple_model model;
    load_model(model, matrix_A_f16, matrix_B_f16, rows_A, cols_A, rows_B, cols_B);

    // perform computation in cpu
    // 注意result是GGML_TYPE_F32类型，在ggml_mul_mat函数中创建
    struct ggml_tensor * result = compute(model);

    // get the result data pointer as a float array to print
    std::vector<float> out_data(ggml_nelements(result));
    memcpy(out_data.data(), result->data, ggml_nbytes(result));

    // expected result:
    // [ 0.60 0.55 0.50 1.10
    //   0.90 0.54 0.54 1.26
    //   0.42 0.29 0.28 0.64 ]

    printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
    for (int j = 0; j < result->ne[1] /* rows */; j++) {
        if (j > 0) {
            printf("\n");
        }

        for (int i = 0; i < result->ne[0] /* cols */; i++) {
            printf(" %.2f", out_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");

    // free memory
    ggml_free(model.ctx);
    return 0;
}
