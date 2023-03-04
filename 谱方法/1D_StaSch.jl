#=
利用谱求导矩阵的一维定态Schrodinger方程求解器:
1、采用原子单位制(a.u.)，定态Schrodinger方程形式为:  -1/2m*▽·(▽ψ) + V*ψ = E*ψ
2、由于谱求导矩阵基于DFT得到，求解边界为周期性边界，势函数V将看作V的周期延拓
3、由于周期性边界影响，束缚态在高能级下的本征态与本征能量计算不准确(除非势场本身就是周期势)
=#
ENV["GKS_ENCODING"] = "utf-8"
using Plots, LinearAlgebra, ToeplitzMatrices
plotlyjs()

# 势函数
function Vfun(x::StepRangeLen)
    V = Vector{Float64}(1/2*x.^2)
    return V
end

# 主函数
function main()
    # 离散设置
    L = 20                        # 求解区域长度
    N = 64                        # 2的整数幂次
    x = L/N*(-N/2:N/2-1)          # 求解区域

    # 构造二阶谱求导矩阵D₂
    h = 2π/N
    vc₀ = [1/2 * (-1)^(j+1) * (csc(j*h/2))^2 for j=1:N-1]
    vc = [-π^2/(3*h^2)-1/6; vc₀]           # D₂ 矩阵的第一列
    D₂ = (2π/L)^2*SymmetricToeplitz(vc)    # 对称常系数矩阵

    # 构建Hamilton矩阵Hami
    m = 1.0     # 电子质量
    Hami = -1/(2*m)*D₂ + diagm(Vfun(x))

    # 求解N个本征值E与各点本征函数值ψ
    S = @time eigen(Hami)
    Val = S.values
    Vec = S.vectors

    # 本征函数归一化
    Vec2 = abs2.(Vec)                      # |ψ|²
    C = sqrt.(1 ./(sum(L/N*Vec2, dims=1))) # 归一化系数
    for j=1:N
        Vec[:, j] = Vec[:, j]*C[j]
    end

    # 绘图
    fig1 = plot(Val, seriestype=:scatter, label="E (a.u.)", xlabel="本征能量序列号 j")
    savefig(fig1, "1D_EigenEnergy.png")
    fig2 = Any[]
    for j in 1:9
        pic = plot(x, Vec[:, j], label=string("E = ", round(Val[j]; digits = 4), " (a.u.)"))
        push!(fig2, pic)
    end

    fig2 = plot(fig2..., layout=(3, 3), xlabel="x (a.u.)", ylabel="ψ", legend=:outertopright)
    savefig(fig2, "1D_EigenFunction.png")
end
main()
