# 对系数矩阵A进行LU分解，并求解线性方程组AX=b
# 输入A、b，返回X
using LinearAlgebra
using SparseArrays
function LinearLU(A::Matrix{Float64}, b::Vector{Float64})   # 指定输入类型能有效减小运算时间
    n = size(A, 2)
    L = sparse(Matrix{Float64}(I, n, n))   # 存储为稀疏格式
    U = sparse(zeros(Float64, n, n))

    U[1, :] = A[1, :]
    L[2:n, 1] = A[2:n, 1]/A[1, 1]
    for k in 2 : n-1
        for j in k : n
            U[k, j] = A[k, j] - sum(L[k, 1:k-1] .* U[1:k-1, j])    # 此处必须为.*
        end
        for i in k+1 : n
            L[i, k] = (A[i, k] - sum(L[i, 1:k-1] .* U[1:k-1, k]))/U[k, k]
        end
    end
    U[n, n] = A[n, n] - sum(L[n, 1:n-1] .* U[1:n-1, n])

    
    L = collect(L)                        # 逆为密集格式，collect转化为密集格式
    U = collect(U)
    #=
    X = inv(U)*(inv(L)*b)
    =#
    
    Y = inv(L) * b
    X = zeros(n, 1)
    X[n] = Y[n] / U[n, n]
    for k in n-1:-1:1
        X[k] = (Y[k] - sum(U[k, k+1:n] .* X[k+1:n])) / U[k, k]      # 回代求解
    end
    
    return X
end
