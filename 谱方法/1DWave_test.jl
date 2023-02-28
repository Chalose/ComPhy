# 测试Fourier谱方法求解一维波动方程初值问题(周期性边界)
using Plots, FFTW, DifferentialEquations
plotlyjs()     # plotlyjs()后端

c = 5.0
# 波矢空间微分方程
function Wave1D!(du, u, k, t)
    du[:, 1] = u[:, 2]
    du[:, 2] = -c^2*k.^2 .* u[:, 1]
end

# 主函数
function main()
    # 离散设置
    L = 80
    N = 256
    x = L/N*(-N/2:N/2-1)
    k = 2π/L*[0:N/2-1; -N/2:-1]

    # 初值条件
    u₀ = 5*exp.(-(x .- 10).^2) .- 10*exp.(-(x .+ 20).^2)
    ut₀ = fft(u₀)                                 # ut, vt指代对u，du/dt作F变换后结果
    vt₀ = zeros(N, 1)                
    uvt₀ = [ut₀ vt₀]

    # 求解
    tspan = (0, 20)                               # t范围要求为长度为2的元组(步长由求解器自动控制)
    prob = ODEProblem(Wave1D!, uvt₀, tspan, k)    # 组装常微分方程问题(方程组，初值，时间范围，参数)
    @time sol = solve(prob, Tsit5()) 

    t = sol.t     # 提取时间步
    uvt = sol.u   # uvt为由length(t)个拼接矩阵[ut vt]构成的结构，uvt[j]指代第j(j=1,2,...,length(t))时间步的[ut vt]

    # 逆变换
    u = zeros(N, length(t))
    @time for j in 1:length(t)
        u[:, j] = real.(ifft(uvt[j][:, 1]))        # 即第j时间步uvt的第1列
    end

    # 绘制gif
    @gif for j in 1:length(t)
        plot(x, u[:, j], title=string("t = ", t[j], "s"), xlims=(-40.5, 40.5), ylims=(-11.5, 6.5))
    end
end
main()
