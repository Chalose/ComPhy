# 采用显式推进计算波动方程传播与反射，求解域为边长为2的正方形，四壁为固定边界(均为0)
#=
                 1
            ————————————
           |           |
           |           |
        3  |           |  4
           |           |
           |           |
            ————————————
                 2

x方向步长dx，y方向步长dy，时间步长dt，有稳定性条件：
4*c^2*dt^2/(dx^2 + dy^2) <= 1  ,  c为波相速度
=#
using Plots
# 求解设置
dx = 0.02        # 保证步数N
dy = 0.02        # 保证M为整数
dt = 0.01
X = [-1.0; 1.0]  # 求解域范围: X[Xmin; Xmax] , Y[Ymin; Ymax]
Y = [-1.0; 1.0]
t = [0.0; 25.0]  # 时间范围t[tmin; tmax]
c = 1.0          # 波相速度

# 求解函数
function Wavesolver_1(dx::Float64, dy::Float64, dt::Float64, X::Vector{Float64}, Y::Vector{Float64}, t::Vector{Float64}, c::Float64)
    # 初始化
    N = Int64((X[2] - X[1])/dx) + 1            # X向步数
    M = Int64((Y[2] - Y[1])/dy) + 1            # Y向步数
    kmax = Int64(ceil((t[2]-t[1])/dt)) + 1     # t步数
    u = zeros(M, N, kmax)
    Px = (c*dt/dx)^2
    Py = (c*dt/dy)^2
    
    # 初值与边界处理
    u[:, :, 1] = [1.0*exp(-((x .- 0.0).^2 .+ (y .- 0.0).^2)/0.005) for y = Y[1]:dy:Y[2], x = X[1]:dx:X[2]]   # 初值条件1
       # 由初值条件2直接插值出t + dt时间步的u(x, y)
    u[1, :, 2] .= 0.0    # 全反射边界条件
    u[M, :, 2] .= 0.0
    u[:, 1, 2] .= 0.0
    u[:, N, 2] .= 0.0
    for j in 2:N-1
        for i in 2:M-1
            u[i, j, 2] = (1-Px-Py)*u[i, j, 1] + Px/2*(u[i, j+1, 1] + u[i, j-1, 1]) + Py/2*(u[i+1, j, 1] + u[i-1, j, 1])
        end
    end

    # 显式推进
    for k in 2:kmax-1
        for j in 2:N-1   # 内点
            for i in 2:M-1
                u[i, j, k+1] = 2*(1-Px-Py)*u[i, j, k] + Px*(u[i, j+1, k] + u[i, j-1, k]) + Py*(u[i+1, j, k] + u[i-1, j, k]) - u[i, j, k-1]
            end
        end
    end

    return u
end

# 后处理
@time u = Wavesolver_1(dx, dy, dt, X, Y, t, c);
Xx = LinRange(X[1]:dx:X[2])
Yy = LinRange(Y[1]:dy:Y[2])

@gif for k in 1:Int64(ceil((t[2]-t[1])/dt)) + 1
    tk = t[1] + (k - 1)*dt
    heatmap(Xx, Yy, u[:, :, k], title = tk )
    #contourf(Xx, Yy, u[:, :, k], title = tk )
end every 5
