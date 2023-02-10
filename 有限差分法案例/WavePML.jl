#=
用于测试非齐次波动方程显式求解以及完美匹配层(PML)对波的吸收
                 1
            ————————————
           |           |
           |           |
        3  |           |  4
           |           |
           |           |
            ————————————
                 2
边界1，2，3，4均为PML
非齐次项为：f₁(x, y)*sin(ω₁t + ϕ₁) + f₂(x, y)*sin(ω₂t + ϕ₂)

           f₁(x, y) = A₁*exp(-b*((x - x₁)^2 + (y - y₁)^2))
           f₂(x, y) = A₂*exp(-b*((x - x₂)^2 + (y - y₂)^2))
           (x₁, y₁)、(x₂, y₂)为两受迫源位置
初值条件为：u(x, y, 0) = 0, du/dt(x, y, 0) = 0
=#
using Plots
using SparseArrays

# 构建网格，配置PML耗散层σ₁(x),σ₂(y)
function BuildMesh(dx::Float64, dy::Float64, dt::Float64, X::Vector{Float64}, Y::Vector{Float64}, 
    T::Vector{Float64}, a::Float64)
    VectorX = Vector{Float64}(X[1]:dx:X[2])
    VectorY = Vector{Float64}(Y[1]:dy:Y[2])
    VectorT = Vector{Float64}(T[1]:dt:T[2]) 
    N = length(VectorX)            # N列
    M = length(VectorY)            # M行
    kmax = length(VectorT)   # 总时间步数kmax

    # 空间网格
    MeshX = zeros(M, N)
    for j = 1:M
        MeshX[j, :] .= VectorX
    end
    MeshY = zeros(M, N)
    for i = 1:N
        MeshY[:, i] .= sort(VectorY, rev = true)
    end

    σ₁ = sparse(zeros(M, N))
    σ₂ = sparse(zeros(M, N))
    λ₁ = Int64(floor(a/dx));  λ₂ = Int64(floor(a/dy));
    s = 50.0  # σ最大值
    # 耗散函数
    f₃(x) = s*((x - (X[1] + a))/(-a) - sin((x - (X[1] + a))/(-a))/(2π))     # 左
    f₄(x) = s*((x - (X[2] - a))/a - sin((x - (X[2] - a))/a)/(2π))           # 右
    f₁(y) = s*((y - (Y[2] - a))/a - sin((y - (Y[2] - a))/a)/(2π))           # 上
    f₂(y) = s*((y - (Y[1] + a))/(-a) - sin((y - (Y[1] + a))/(-a))/(2π))     # 下
    # 配置PML耗散层σ₁(x),σ₂(y)
    σ₁[:, 1:λ₁] = map(f₃, MeshX[:, 1:λ₁])
    σ₁[:, N-(λ₁-1):N] = map(f₄, MeshX[:, N-(λ₁-1):N])
    σ₂[1:λ₂, :] = map(f₁, MeshY[1:λ₂, :])
    σ₂[M-(λ₂-1):M, :] = map(f₂, MeshY[M-(λ₂-1):M, :])

    return (N, M, kmax, MeshX, MeshY, VectorT, σ₁, σ₂)
end

# 含时非齐次项处理
function Source(i::Int64, j::Int64, k::Int64, σ₁::SparseMatrixCSC{Float64, Int64}, 
    σ₂::SparseMatrixCSC{Float64, Int64}, MeshX::Matrix{Float64}, MeshY::Matrix{Float64}, 
    VectorT::Vector{Float64})
    # 受迫源设置
    ω₁ = 2π*10.0     # 受迫源1角频率
    ω₂ = 2π*10.0     # 受迫源2角频率
    ϕ₁ = 0.0         # 受迫源1初相
    ϕ₂ = 3.14159265         # 受迫源2初相
    A₁ = 50.0        # 受迫源1最大幅值
    A₂ = 50.0        # 受迫源2最大幅值
    x₁ = -0.25;  y₁ = 0.0;     # 受迫源1位置
    x₂ = 0.25;   y₂ = 0.0;     # 受迫源2位置

    x = MeshX[i, j]
    y = MeshY[i, j]
    t = VectorT[k]
    f₁(x, y) = A₁*exp(-((x - x₁)^2 + (y - y₁)^2)/0.005)
    f₂(x, y) = A₂*exp(-((x - x₂)^2 + (y - y₂)^2)/0.005)
    F = f₁(x, y)*sin(ω₁*t + ϕ₁) + f₂(x, y)*sin(ω₂*t + ϕ₂) -
        sqrt(2π)*σ₁[i, j]*σ₂[i, j]*(f₁(x, y)*sin(ω₁*t + ϕ₁)/ω₁^2 + f₂(x, y)*sin(ω₂*t + ϕ₂)/ω₂^2) - 
        sqrt(2π)*(σ₁[i, j]+σ₂[i, j])*(f₁(x, y)*cos(ω₁*t + ϕ₁)/ω₁ + f₂(x, y)*cos(ω₂*t + ϕ₂)/ω₂)
    
    return F
end

# 显式求解
function PMLtest(N::Int64, M::Int64, kmax::Int64, σ₁::SparseMatrixCSC{Float64, Int64}, 
    σ₂::SparseMatrixCSC{Float64, Int64}, c::Float64, dx::Float64, dy::Float64, dt::Float64,
    MeshX::Matrix{Float64}, MeshY::Matrix{Float64}, VectorT::Vector{Float64})
    # 初值条件
    u = zeros(M, N, kmax)
    #Du = zeros(M, N, kmax)
    v₁ = zeros(M, N, kmax)    # 辅助变量v₁，v₂  (由于u初值为0，故梯度为0，故辅助变量初值为0)
    v₂ = zeros(M, N, kmax)

    # 显式推进
    P₁(i, j) = (σ₁[i, j]+σ₂[i, j])*dt/2
    P₂(i, j) = σ₁[i, j]*σ₂[i, j]*dt^2
    P₃ = (c*dt/dx)^2
    P₄ = (c*dt/dy)^2
    P₅ = (c*dt)^2/(2*dx)
    P₆ = (c*dt)^2/(2*dy)
    # k=2 层
    for j in 2:N-1
        for i in 2:M-1
            u[i, j, 2] = ((1-P₃-P₄) - P₂(i, j)/2)*u[i, j, 1] + P₃/2*(u[i, j+1, 1] + u[i, j-1, 1]) + 
                        P₄/2*(u[i+1, j, 1] + u[i-1, j, 1]) + P₅/2*(v₁[i, j+1, 1] - v₁[i, j-1, 1]) + 
                        P₆/2*(v₂[i+1, j, 1] - v₂[i-1, j, 1]) + dt^2/2*Source(i, j, 2, σ₁, σ₂, MeshX, MeshY, VectorT)
            
            v₁[i, j, 2] = (1 - dt*σ₁[i, j])*v₁[i, j, 1] + dt*(σ₂[i, j]-σ₁[i, j])/(2*dx)*(u[i, j+1, 1] - u[i, j-1, 1])
            v₂[i, j, 2] = (1 - dt*σ₂[i, j])*v₂[i, j, 1] + dt*(σ₁[i, j]-σ₂[i, j])/(2*dy)*(u[i+1, j, 1] - u[i-1, j, 1])
        end
    end
    # k=3:kmax层
    for k in 3:kmax
        for j in 2:N-1
            for i in 2:M-1 
                u[i, j, k] = 1/(1 + P₁(i, j))*((2*(1-P₃-P₄) - P₂(i, j))*u[i, j, k-1] + P₃*(u[i, j+1, k-1] + u[i, j-1, k-1]) + 
                            P₄*(u[i+1, j, k-1] + u[i-1, j, k-1]) + (P₁(i, j) - 1)*u[i, j, k-2] + P₅*(v₁[i, j+1, k-1] - v₁[i, j-1, k-1]) + 
                            P₆*(v₂[i+1, j, k-1] - v₂[i-1, j, k-1]) + dt^2*Source(i, j, k, σ₁, σ₂, MeshX, MeshY, VectorT))
                
                v₁[i, j, k] = (1 - dt*σ₁[i, j])*v₁[i, j, k-1] + dt*(σ₂[i, j]-σ₁[i, j])/(2*dx)*(u[i, j+1, k-1] - u[i, j-1, k-1])
                v₂[i, j, k] = (1 - dt*σ₂[i, j])*v₂[i, j, k-1] + dt*(σ₁[i, j]-σ₂[i, j])/(2*dy)*(u[i+1, j, k-1] - u[i-1, j, k-1])
            end
        end
    end

    return u
end
#######################################################################################################
# 主函数
function main()
    # 求解设置
    dx = 0.01
    dy = 0.01
    dt = 0.005
    X = [-1.0; 1.0]  # 求解域范围: X[Xmin; Xmax] , Y[Ymin; Ymax]
    Y = [-1.0; 1.0]
    T = [0.0; 3.0]   # 时间范围：T[tmin; tmax]
    c = 1.0          # 波相速度
    a = 0.14         # 完美匹配层厚度a
    # 网格建立
    (N, M, kmax, MeshX, MeshY, VectorT, σ₁, σ₂) = BuildMesh(dx, dy, dt, X, Y, T, a)

    # 方程求解
    u = @time PMLtest(N, M, kmax, σ₁, σ₂, c, dx, dy, dt, MeshX, MeshY, VectorT)

    # 后处理
    Xx = LinRange(X[1]:dx:X[2])
    Yy = LinRange(Y[1]:dy:Y[2])
    @gif for k in 1:kmax
        tk = T[1] + (k - 1)*dt
        contourf(Xx, Yy, u[:, :, k], title=tk, color=:turbo, size=(600, 500))
    end

end
main()
