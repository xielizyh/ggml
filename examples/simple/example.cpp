#include "ggml.h"
#include "ggml-cpu.h"

// f(x)=ax^2 + b

int main(void)
{
    struct ggml_init_params params = {
        /*.mem_size   =*/ 16*1024*1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };

    // memory allocation happens here
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_param(ctx, x); // x is an input variable

    struct ggml_tensor * a  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * b  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * x2 = ggml_mul(ctx, x, x);
    struct ggml_tensor * f  = ggml_add(ctx, ggml_mul(ctx, a, x2), b);

    /* 计算 */
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, f);

    // set the input variable and parameter values
    ggml_set_f32(x, 2.0f);
    ggml_set_f32(a, 3.0f);
    ggml_set_f32(b, 4.0f);

    ggml_graph_compute_with_ctx(ctx, gf, 1);

    printf("f = %f\n", ggml_get_f32_1d(f, 0));

    return 0;
}