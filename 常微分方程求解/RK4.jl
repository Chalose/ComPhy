# 采用4阶精度龙格库塔(Runge-Kutta)方法求解一阶常微分方程组初值问题
# 输入一阶方程组函数f，初值向量Y0，范围xspan(a, b)，步数n
# 输出自变量取值向量X，解矩阵sol，不同列指不同y随自变量的演化
function RK4(f::Function, Y0::Vector{Float64}, xspan::Tuple{Float64, Float64}, n::Int64)
    N = size(Y0)
    K1 = zeros(N); K2 = K1; K3 = K1; K4 = K1;
    Y = Y0
    h = (xspan[2] - xspan[1])/n
    X = Array{Float64}(xspan[1] : h : xspan[2])
    sol = zeros(n+1, N[1]); sol[1, :] = Y0

    # 4阶R-K推进
    for i in 1: n
        K1 = f(X[i], Y)
        K2 = f(X[i] + h/2, Y + h/2*K1)
        K3 = f(X[i] + h/2, Y + h/2*K2)
        K4 = f(X[i] + h, Y + h*K3)
        Y = Y + h/6*(K1 + 2K2 + 2K3 + K4)
        sol[i+1, :] = Y
    end

    return X, sol
end
