import torch
import time
import linear_operator.operators.kronecker_product_linear_operator

orig_matmul = linear_operator.operators.kronecker_product_linear_operator._matmul
orig_t_matmul = linear_operator.operators.kronecker_product_linear_operator._t_matmul

KronMatmulTime = 0
def new_matmul(linear_ops, kp_shape, rhs):
    global KronMatmulTime
    torch.cuda.synchronize()
    s = time.time()
    res = orig_matmul(linear_ops, kp_shape, rhs)
    torch.cuda.synchronize()
    e = time.time()
    KronMatmulTime += (e - s) * 1000
    return res

def new_t_matmul(linear_ops, kp_shape, rhs):
    global KronMatmulTime
    torch.cuda.synchronize()
    s = time.time()
    res = orig_t_matmul(linear_ops, kp_shape, rhs)
    torch.cuda.synchronize()
    e = time.time()
    KronMatmulTime += (e - s) * 1000
    return res

linear_operator.operators.kronecker_product_linear_operator._matmul = new_matmul
linear_operator.operators.kronecker_product_linear_operator._t_matmul = new_t_matmul