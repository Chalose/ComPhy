#= 基于SOR点迭代的Posiion方程边值问题求解(二阶精度)

正方形域： -1<=x<=1 , -1<=y<=1
▽·(▽u) = -2π^2sin(πx)cos(πy)
u(-1, y) = u(1, y) = 0
dudy(x, -1) = dudy(x, 1) = 0
=#
using Plots

# 求解设置
M = 100               # 每行节点数
h = 2/(M - 1)         # 步长
eps = 1e-8            # 迭代残差限
f = [-2pi^2*sin(pi*x)*cos(pi*y) for y = 1:-h:-1, x = -1:h:1]    # 源项

# 求解函数
function SORPossion(M::Int64, h::Float64, eps::Float64, f::Matrix{Float64})
    # 初始化
    u = zeros(M, M)
    u0 = zeros(M, M)
    kmax = Int64(5e3)              # 最大迭代次数
    epsk = zeros(kmax, 1)          # k迭代残差
    w = 2/(1 + sin(pi/M))          # 松弛因子
    k = 1                          # 迭代次数计数

    # 迭代
    while true
        # 内点
        for j in 2:M-1
            for i in 2:M-1
                u[i, j] = (1-w)*u[i, j] + w/4*(u[i, j+1] + u[i, j-1] + u[i+1, j] + u[i-1, j] - h^2*f[i, j])     # 内点
            end
        end
        # 左右边界(第一类边界条件)
        #u[:, 1] .= 0.0     # 注意点运算(对于初始零矩阵，可不加)
        #u[:, M] .= 0.0
        # 上下边界(第二类边界条件)
        #=
        u[1, 2:M-1] .= [1/4*(u[1, j+1] + u[1, j-1] + 2u[2, j] - h^2*f[1, j]) for j =2:M-1]
        u[M, 2:M-1] .= [1/4*(u[M, j+1] + u[M, j-1] + 2u[M-1, j] - h^2*f[M, j]) for j =2:M-1]
        =#
        for j in 2:M-1
            u[1, j] = 1/4*(u[1, j+1] + u[1, j-1] + 2*u[2, j] - h^2*f[1, j])
            u[M, j] = 1/4*(u[M, j+1] + u[M, j-1] + 2*u[M-1, j] - h^2*f[M, j])
        end
        # 迭代残差
        epsk[k] = maximum(abs.(u - u0))
        if epsk[k]<eps && k>10
            break
        elseif k>=kmax
            print("达到最大迭代次数")
            break
        end
        u0 = copy(u)                     # 保存上一步迭代结果
        k += 1
    end

    return (u, epsk)
end

# 后处理
@time (u, epsk) = SORPossion(M, h, eps, f)

X = LinRange(-1:h:1)
Y = LinRange(-1:h:1)
contourf(X, Y, u)
#savefig("possion1.png")
#=
plot(epsk[1:2652], 
     lw = 1.5,
     xlabel = "Steps",
     ylabel = "Eps",
     label = "SOR")
     =#
#savefig("SOR_eps.png")
