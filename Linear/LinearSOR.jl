# SOR迭代求解线性方程组AX=b
# 给定A、b、残差限δ，迭代初值默认为零矢量;返回X、迭代残差detak
# w = 1即Gauss-Seidel迭代
function LinearSOR(A::Matrix{Float64}, b::Vector{Float64}, deta::Float64)
    n = size(A, 2)
    X = zeros(Float64, n, 1)
    Xk = copy(X)
    detak = 0.0
    w = 1.1    # 松弛因子

    k = 1
    while true
        for i in 1:n
            Xk[i] = X[i] + w/A[i, i] *(b[i] - sum(A[i, 1:i-1] .* Xk[1:i-1]) - sum(A[i, i:n] .* X[i:n]))
        end
        D = Xk - X
        detak = sum(D[1:n].^2)^0.5

        if k>10 && detak<deta
            break
        elseif k>1e4
            print("Error:超出最大迭代次数")
            break
        end
        X = copy(Xk)
        k += 1
    end

    return (X, detak)
end
