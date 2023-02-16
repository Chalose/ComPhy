#=
可对单电子1D含时Schrödinger方程进行求解，采用Crank-Nicholson方法；
边界条件形式：a*Ψ + b*dΨ/dn = c(t)；  左1右2
采取原子单位制(a.u.)；
=#
using Plots
using SparseArrays
using LinearAlgebra

# 势函数
function Vfun(i::Int64, k::Int64, dx::Float64, dt::Float64, X::Vector{Float64}, T::Vector{Float64})
    # 这里采用定常方势垒，也可以设置含时势
    a = 1.0
    b = 2.0
    V₀ = 50.0
    #t = T[1] + (k-1)*dt
    x = X[1] + (i-1)*dx
    if a<=x<=b
        return V₀
    else
        return 0.0
    end
end

# 隐式求解
function Solver_1D(dx::Float64, dt::Float64, X::Vector{Float64}, T::Vector{Float64})
    # 构建网格
    VectorX = Vector{Float64}(X[1]:dx:X[2])
    VectorT = Vector{Float64}(T[1]:dt:T[2])
    N = length(VectorX)
    kmax = length(VectorT)

    # 初值条件
    ψ = Array{ComplexF64}(zeros(N, kmax))
    A₀ = 0.75
    a = 35.0
    ω = 5.0
    c = 1.0;  k = ω/c
    ψ[:, 1] = A₀*exp.(-a*(VectorX .- 0.25).^2).*exp.(im*(k*VectorX))   # Gauss波包

    # 边界设置
    a = [1.0; 1.0]   # Dirichlet boundary: Ψ=0
    b = [0.0; 0.0]
    c = zeros(2, kmax)  # 第一、二行分别为c₁、c₂随时间演化

    # Crank-Nicholson method
    P = -im*dt/(4*dx^2)
    R = -P
    if b == [0.0; 0.0]  # 针对Dirichlet boundary
        λ₁ = 1.0;  λ₂ = 0.0;  λ₃ = 0.0;  λ₄ = 1.0;
        λ₅ = 0.0;  λ₆ = 0.0;  λ₇ = 0.0;  λ₈ = 0.0;

        L₁ = P*[1.0 for i = 1:N-1];
        U₁ = copy(L₁);
        L₂ = R*[1.0 for i = 1:N-1];
        U₂ = copy(L₂); 
        L₁[N-1] = λ₃;  U₁[1] = λ₂;
        L₂[N-1] = λ₇;  U₂[1] = λ₆; 
        E = sparse(zeros(N, kmax));  E[1, :] .= c[1, :];  E[N, :] .= c[2, :];
        for k in 2:kmax
            # 组装求解矩阵A、B
            D₁ = [1 + im*dt/2*Vfun(i, k, dx, dt, X, T) + im*dt/(2*dx^2) for i = 1:N]
            D₂ = [1 - im*dt/2*Vfun(i, k-1, dx, dt, X, T) - im*dt/(2*dx^2) for i = 1:N]
            D₁[1] = λ₁
            D₁[N] = λ₄
            D₂[1] = λ₅ 
            D₂[N] = λ₈
            A = sparse(Tridiagonal(L₁, D₁, U₁))
            B = sparse(Tridiagonal(L₂, D₂, U₂))
            # Thomas法求解
            Vec = B*ψ[:, k-1] + E[:, k]
            ψ[:, k] = TMas(A, Vec)
        end

    else   # 其他边界情况
        λ₂ = 2*P;  λ₃ = 2*P;
        λ₆ = 2*R;  λ₇ = 2*R;

        L₁ = P*[1.0 for i = 1:N-1];
        U₁ = copy(L₁);
        L₂ = R*[1.0 for i = 1:N-1];
        U₂ = copy(L₂); 
        L₁[N-1] = λ₃;  U₁[1] = λ₂;
        L₂[N-1] = λ₇;  U₂[1] = λ₆;
        E = sparse(zeros(N, 1));
        for k in 2:kmax
            D₁ = [1 + im*dt/2*Vfun(i, k, dx, dt, X, T) + im*dt/(2*dx^2) for i = 1:N]
            D₂ = [1 - im*dt/2*Vfun(i, k-1, dx, dt, X, T) - im*dt/(2*dx^2) for i = 1:N]
            D₁[1] = (1 + im*dt/2*Vfun(1, k, dx, dt, X, T) + im*dt/(2*dx^2)) - 2*dx*P*a[1]/b[1]
            D₁[N] = (1 + im*dt/2*Vfun(N, k, dx, dt, X, T) + im*dt/(2*dx^2)) - 2*dx*P*a[2]/b[2]
            D₂[1] = (1 - im*dt/2*Vfun(1, k-1, dx, dt, X, T) - im*dt/(2*dx^2)) - 2*dx*R*a[1]/b[1]
            D₂[N] = (1 - im*dt/2*Vfun(N, k-1, dx, dt, X, T) - im*dt/(2*dx^2)) - 2*dx*R*a[2]/b[2]
            A = sparse(Tridiagonal(L₁, D₁, U₁))
            B = sparse(Tridiagonal(L₂, D₂, U₂))
            E[1] = 2*dx/b[1]*(R*c[1, k-1] - P*c[1, k])
            E[N] = 2*dx/b[2]*(R*c[2, k-1] - P*c[2, k])
            # Thomas法求解
            Vec = B*ψ[:, k-1] + E
            ψ[:, k] = TMas(A, Vec)
        end
    end

    return ψ
end

# Thomas法求解器
function TMas(A::SparseMatrixCSC{ComplexF64, Int64}, R::SparseVector{ComplexF64, Int64})
    U = Bidiagonal(A, :U)                   # 生成上双对角矩阵
    n = size(A, 2)
    y = Array{ComplexF64}(zeros(n, 1))

    U[1, 1] = A[1, 1]
    y[1] = R[1]
    for i in 2:n                            # 追过程(消元)
        U[i, i] = A[i, i] - A[i, i-1]/U[i-1, i-1] * A[i-1, i]
        y[i] = R[i] - A[i, i-1]/U[i-1, i-1] * y[i-1]  
    end

    X = Array{ComplexF64}(zeros(n, 1))      # 赶过程(回代)
    X[n] = y[n]/U[n, n]
    for i in n-1:-1:1
        X[i] = (y[i] - A[i, i+1] * X[i+1])/U[i, i]
    end

    return X
end

###################################################################################################3
# 主函数
function main()
    # 设置
    dx = 0.01
    dt = 0.001
    X = [-1.0; 4.0]   # 求解域范围: X[Xmin; Xmax]
    T = [0.0; 1.0]    # 时间范围：T[tmin; tmax]
    # 求解
    ψ = @time Solver_1D(dx, dt, X, T)
    Re_ψ = real.(ψ)
    Im_ψ = imag.(ψ)
    Abs2_ψ = abs2.(ψ)
    # 绘图
    Xx = LinRange(X[1]:dx:X[2])
    V = [Vfun(i, 1, dx, dt, X, T) for i=1:length(Xx)]
    Tt = LinRange(T[1]:dt:T[2])
    @gif for k in 1:length(Tt)
        tk = T[1] + (k - 1)*dt
        tk = string("t = ", tk, " (a.u.)")
        plot(Xx, Abs2_ψ[:, k],
        xlabel="x (a.u.)", 
        ylabel="|ψ|²", 
        ylims=(-0.05, 1.0), 
        title=tk, 
        label="Electron", 
        lengend=:topleft)

        plot!(twinx(), Xx, V, 
        ylabel="E (a.u.)",
        color=:red,
        label="Potential function", 
        lengend=:topright)
    end
end
main()
