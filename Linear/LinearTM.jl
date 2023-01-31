# 追赶法(Thomas法)求解线性方程组AX=d
# 要求A为三对角矩阵
using LinearAlgebra
using SparseArrays
function LinearTM(A::Matrix{Float64}, d::Vector{Float64})
    U = sparse(Bidiagonal(A, :U))           # 生成上双对角矩阵，存储为稀疏矩阵
    n = size(A, 2)
    y = zeros(n, 1)

    U[1, 1] = A[1, 1]
    y[1] = d[1]
    for i in 2:n                            # 追过程(消元)
        U[i, i] = A[i, i] - A[i, i-1]/U[i-1, i-1] * A[i-1, i]
        y[i] = d[i] - A[i, i-1]/U[i-1, i-1] * y[i-1]  
    end

    X = zeros(n, 1)                         # 赶过程(回代)
    X[n] = y[n]/U[n, n]
    for i in n-1:-1:1
        X[i] = (y[i] - A[i, i+1] * X[i+1])/U[i, i]
    end

    return X
end
