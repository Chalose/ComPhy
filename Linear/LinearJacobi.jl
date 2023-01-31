# 利用Jacobi迭代求解线性方程组
# 给定A、b、残差限δ，迭代初值默认为零矢量;返回X、迭代残差detak
# Jacobi迭代收敛条件为矩阵B的谱半径<1
function LinearJacobi(A::Matrix{Float64}, b::Vector{Float64}, deta::Float64)
    n = size(A, 2)
    B = zeros(Float64, n, n)
    X = zeros(Float64, n, 1);  Xk = copy(X);  F = copy(X);  D = copy(X);   # x存放上一次迭代值；Xk为当前迭代值；D为残差向量
    detak = zeros(Float64, Int64(1e2+1), 1)                       # 保存每次迭代残差

    for i in 1:n                # 构造Jacobi迭代矩阵B, F
        for j in 1:n 
            if i != j
                B[i, j] = -A[i, j]/A[i, i]
            end
        end 
        F[i] = b[i]/A[i, i]
    end

    k = 1                       # 迭代过程         
    while true
        Xk = B * X + F
        D = Xk - X
        detak[k] = sum(D[1:n].^2)^0.5
        if k>10 && detak[k]<deta
            break
        elseif k>1e4            # 最大迭代次数限制
            print("Error:超出最大迭代次数")
            break
        end
        X = copy(Xk)
        k += 1
    end

    return (X, detak)
end
