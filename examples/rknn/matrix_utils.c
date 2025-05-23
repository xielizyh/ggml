#include <stdio.h>

static void matrix_print(const char* name, const float* matrix, int rows, int cols)
{
    printf("Matrix %s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%6.3f ", (double)matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

static void matrix_transpose(const float* src, float* dst, int rows, int cols)
{
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

static void matrix_pad(const float* src, float* dst, int rows, int cols, int pad_rows, int pad_cols)
{
    for (int i = 0; i < pad_rows; ++i) {
        for (int j = 0; j < pad_cols; ++j) {
            if (i < rows && j < cols) {
                dst[i * pad_cols + j] = src[i * cols + j];
            } else {
                dst[i * pad_cols + j] = 0.0f;
            }
        }
    }
}

static void matrix_unpad(const float* src, float* dst, int rows, int cols, int unpad_rows, int unpad_cols)
{
    for (int i = 0; i < unpad_rows; ++i) {
        for (int j = 0; j < unpad_cols; ++j) {
            dst[i * unpad_cols + j] = src[i * cols + j];
        }
    }
}

// 先填充再转置：pad_rows原始矩阵要填充的行数，pad_cols原始矩阵要填充的列数
static void matrix_pad_and_transpose(const float* src, float* dst, int rows, int cols, int pad_rows, int pad_cols)
{
    for (int i = 0; i < pad_rows; ++i) {
        for (int j = 0; j < pad_cols; ++j) {
            if (i < rows && j < cols) {
                dst[j * pad_rows + i] = src[i * cols + j];
            } else {
                dst[j * pad_rows + i] = 0.0f;
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

static void matrix_multiply(const float* a, const float* b, float* c, int m, int n, int k)
{
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < k; ++j) {
            c[i * k + j] = 0.0f;
            for (int l = 0; l < n; ++l) {
                c[i * k + j] += a[i * n + l] * b[l * k + j];
            }
        }
    }
}

int main(int argc, char* argv[])
{
    #define A_ROWS 2
    #define A_COLS 3
    float matrix_A[A_ROWS*A_COLS] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
    #define B_ROWS 3
    #define B_COLS 4
    const float matrix_B[B_ROWS*B_COLS] = {
        7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0,
        15.0, 16.0, 17.0, 18.0
    };

    matrix_print("A", matrix_A, A_ROWS, A_COLS);
    matrix_print("B", matrix_B, B_ROWS, B_COLS);

    /* 转置 */
    float matrix_A_T[A_COLS*A_ROWS];
    matrix_transpose(matrix_A, matrix_A_T, A_ROWS, A_COLS);
    matrix_print("A^T", matrix_A_T, A_COLS, A_ROWS);

    /* 填充 */
    #define PAD_ROWS 4
    #define PAD_COLS 5
    float matrix_A_pad[PAD_ROWS*PAD_COLS];
    matrix_pad(matrix_A, matrix_A_pad, A_ROWS, A_COLS, PAD_ROWS, PAD_COLS);
    matrix_print("A_pad", matrix_A_pad, PAD_ROWS, PAD_COLS);
    
    /* 去填充 */
    float matrix_A_unpad[A_ROWS*A_COLS];
    matrix_unpad(matrix_A_pad, matrix_A_unpad, PAD_ROWS, PAD_COLS, A_ROWS, A_COLS);
    matrix_print("A_unpad", matrix_A_unpad, A_ROWS, A_COLS);
    
    /* 填充+转置 */
    float matrix_A_T_pad[PAD_COLS*PAD_ROWS];
    matrix_pad_and_transpose(matrix_A, matrix_A_T_pad, A_ROWS, A_COLS, PAD_ROWS, PAD_COLS);
    matrix_print("A^T_pad", matrix_A_T_pad, PAD_COLS, PAD_ROWS);
    
    /* 去填充+转置 */
    float matrix_A_T_unpad[A_COLS*A_ROWS];
    matrix_unpad_and_transpose(matrix_A_T_pad, matrix_A_T_unpad, PAD_COLS, PAD_ROWS, A_COLS, A_ROWS);
    matrix_print("A_unpad^T", matrix_A_T_unpad, A_ROWS, A_COLS);
    
    /* 乘法 */
    #define C_ROWS A_ROWS
    #define C_COLS B_COLS
    float matrix_C[C_ROWS*C_COLS];
    matrix_multiply(matrix_A_T_unpad, matrix_B, matrix_C, A_COLS, A_ROWS, B_COLS);
    matrix_print("C=A*B", matrix_C, C_ROWS, C_COLS);

    return 0;
}