# 测试Fourier谱方法求解二维波动方程初值问题(周期性边界)
using Plots, FFTW, DifferentialEquations
plotlyjs()

c = 1.0
function Wave2D!(du, u, K2, t)
    du[:, 1] = u[:, 2]
    du[:, 2] = -c^2*K2 .* u[:, 1]
end

function main()
    # 离散设置
    L = 4
    N = 128
    x = L/N*(-N/2:N/2-1)        # 右为x正向，下为y正向
    y = copy(x)
    kx = 2π/L*[0:N/2-1; -N/2:-1]
    ky = copy(kx)

    Mesh_X = x' .* ones(N)      # 相当于直积
    Mesh_Y = ones(N)' .* y
    Mesh_kx = kx' .* ones(N)
    Mesh_ky = ones(N)' .* ky
    K2 = Mesh_kx.^2 + Mesh_ky.^2

    # 初值条件
    u₀ = exp.(-20*((Mesh_X .- 0.4).^2 .+ (Mesh_Y .+ 0.4).^2)) .+ exp.(-20*((Mesh_X .+ 0.4).^2 .+ (Mesh_Y .- 0.4).^2))
    ut₀ = fft(u₀)
    vt₀ = zeros(N, N)
    uvt₀ = [ut₀[:] vt₀[:]]  # 得到一个(N²x2) 的矩阵，第一列为ut₀，第二列为vt₀，每列按列存储顺序排列

    # 求解
    tspan = (0, 1)
    prob = ODEProblem(Wave2D!, uvt₀, tspan, K2[:])
    @time sol = solve(prob, Tsit5())
    t = sol.t     # 提取时间步
    uvt = sol.u   # [ut vt]时间序列

    # 逆变换
    u = zeros(N, N, length(t))
    @time for j in 1:length(t)
        u[:, :, j] = real.(ifft(reshape(uvt[j][:, 1], N, N)))   # 还原为矩阵形式(注意ifft和reshape的先后顺序)
    end

    # 绘图
    surface(x, y, u[:, :, 30], title=string("t = ", t[30], "s"), zlims=(-0.5, 0.5), clim=(-0.5, 0.5))
    #=
    @gif for j in 1:length(t)
        surface(x, y, u[:, :, j], title=string("t = ", t[j], "s"), zlims=(-0.5, 0.5), clim=(-0.5, 0.5))
    end
    =#
end
main()
