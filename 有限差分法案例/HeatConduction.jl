# ADI格式求解二维热传导定解问题(扩散方程定解问题)，无条件稳定
# 长方形域，x(-25~25)，y(0~25)，顶边有温度分布为高斯型，其余3边绝热，初始温度为0
#=
        1
    ----------
    |        |
 3  |        |  4
    |        |
    ----------
        2

边界条件：  a*u + b*du/dn = c(x, y, t)
=#
using Plots
using LinearAlgebra
using SparseArrays
# 求解设置
h = 0.25              # 空间步长(保证步数N, M为整数)
dt = 0.1              # 时间步长
X = [-25.0; 25.0]     # 求解域范围: X[Xmin; Xmax] , Y[Ymin; Ymax]
Y = [0.0; 25.0]
t = [0.0; 100.0]      # 时间范围t[tmin; tmax]
mu = 0.5              # 热扩散系数

# 求解函数
function ADIdiffusion(h::Float64, dt::Float64, X::Vector{Float64}, Y::Vector{Float64}, t::Vector{Float64}, D::Float64)
    # 初始化
    N = Int64((X[2] - X[1])/h) + 1            # X向步数
    M = Int64((Y[2] - Y[1])/h) + 1            # Y向步数
    kmax = Int64(ceil((t[2]-t[1])/dt)) + 1    # t步数
    u = zeros(M, N, kmax)                     # 存储各时间步数据

    # 初值与边界设置
    u[:, :, 1] = zeros(M, N)     # 初值条件
    a = [1.0; 0.0; 0.0; 0.0]     # 边界条件系数a1=1,a234=0
    b = [0.0; 1.0; 1.0; 1.0]     # 边界条件系数b1=0,b234=1    

    alpha = 100.0                # 最高加热端温度
    beta = 2.5                  # 控制加热范围
    c1 = zeros(1, N)
    c1[1, :] .= [alpha*exp(-1/(2*beta^2) * x.^2) for x = X[1]:h:X[2]]   # 1边界高斯加热
    c2 = 0.0;  c3 = 0.0;  c4 = 0.0;                                     # 2，3，4边界绝热

    # 交替方向解向量ux,uy与系数矩阵Ax,Ay
    P = 2dt*D/h^2
    P1 = 1/P + 1
    P2 = 1/P - 1
    Rx = zeros(N, 1)             # x向刷新非齐次项
    Ry = zeros(M, 1)             # y向刷新非齐次项

    Lx = [-1.0 for i = 1:N-1];  Lx[N-1] = -2*b[4];
    Dx = [2*P1 for i = 1:N];  Dx[1] = 2*(b[3]*P1 + h*a[3]);  Dx[N] = 2*(b[4]*P1 + h*a[4]);
    Ux = [-1.0 for i = 1:N-1];  Ux[1] = -2*b[3];
    Ax = sparse(Tridiagonal(Lx, Dx, Ux))

    Ly = [-1.0 for i = 1:M-1];  Ly[M-1] = -2*b[2];
    Dy = [2*P1 for i = 1:M];  Dy[1] = 2*(b[1]*P1 + h*a[1]);  Dy[M] = 2*(b[2]*P1 + h*a[2]);
    Uy = [-1.0 for i = 1:M-1];  Uy[1] = -2*b[1];
    Ay = sparse(Tridiagonal(Ly, Dy, Uy))
    
    # 交替方向求解
    for k in 2:kmax
        if mod(k, 2) == 0    # k为偶数，作x向刷新
            u[1, :, k] .= [1/(2*h*a[1] + 3b[1]) * (2*h*c1[j] + 4*b[1]*u[2, j, k-1] - b[1]*u[3, j, k-1]) for j = 1:N]    # 1行
            u[M, :, k] .= [1/(2*h*a[2] + 3b[2]) * (2*h*c2 + 4*b[2]*u[M-1, j, k-1] - b[2]*u[M-2, j, k-1]) for j = 1:N]   # M行
            
            for i in 2:M-1
                Rx[1] = b[3]*(u[i+1, 1, k-1] + 2*P2*u[i, 1, k-1] + u[i-1, 1, k-1]) + 2*h*c3
                Rx[2:N-1] .= [u[i+1, j, k-1] + 2*P2*u[i, j, k-1] + u[i-1, j, k-1] for j = 2:N-1]
                Rx[N] = b[4]*(u[i+1, N, k-1] + 2*P2*u[i, N, k-1] + u[i-1, N, k-1]) + 2*h*c4

                u[i, :, k] .= TMas(Ax, Rx)
            end
        else                 # k为奇数，y向刷新
            u[:, 1, k] .= [1/(2*h*a[3] + 3b[3]) * (2*h*c3 + 4*b[3]*u[i, 2, k-1] - b[3]*u[i, 3, k-1]) for i = 1:M]      # 1列
            u[:, N, k] .= [1/(2*h*a[4] + 3b[4]) * (2*h*c4 + 4*b[4]*u[i, N-1, k-1] - b[4]*u[i, N-2, k-1]) for i = 1:M]  # N列
            
            for j in 2:N-1 
                Ry[1] = b[1]*(u[1, j+1, k-1] + 2*P2*u[1, j, k-1] + u[1, j-1, k-1]) + 2*h*c1[j]
                Ry[2:M-1] .= [u[i, j+1, k-1] + 2*P2*u[i, j, k-1] + u[i, j-1, k-1] for i = 2:M-1]
                Ry[M] = b[2]*(u[M, j+1, k-1] + 2*P2*u[M, j, k-1] + u[M, j-1, k-1]) + 2*h*c2

                u[:, j, k] .= TMas(Ay, Ry)
            end
        end
    end

    return u
end

# Thomas法求解器
function TMas(A::SparseMatrixCSC{Float64, Int64}, R::Matrix{Float64})
    U = Bidiagonal(A, :U)           # 生成上双对角矩阵
    n = size(A, 2)
    y = zeros(n, 1)

    U[1, 1] = A[1, 1]
    y[1] = R[1]
    for i in 2:n                            # 追过程(消元)
        U[i, i] = A[i, i] - A[i, i-1]/U[i-1, i-1] * A[i-1, i]
        y[i] = R[i] - A[i, i-1]/U[i-1, i-1] * y[i-1]  
    end

    X = zeros(n, 1)                         # 赶过程(回代)
    X[n] = y[n]/U[n, n]
    for i in n-1:-1:1
        X[i] = (y[i] - A[i, i+1] * X[i+1])/U[i, i]
    end

    return X
end

# 后处理
T = @time ADIdiffusion(h, dt, X, Y, t, mu)
X = LinRange(X[1]:h:X[2])
Y = LinRange(Y[1]:h:Y[2])

@gif for k in 1:Int64(ceil((t[2]-t[1])/dt)) + 1
    tk = t[1] + (k - 1)*dt
    contourf(X, Y, T[:, :, k], title = tk )
end every 5
