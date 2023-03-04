#=
利用谱求导矩阵的二维定态Schrodinger方程求解器:
1、采用原子单位制(a.u.)，定态Schrodinger方程形式为:  -1/2m*▽·(▽ψ) + V*ψ = E*ψ
2、由于谱求导矩阵基于DFT得到，求解边界为周期性边界，势函数V将看作V的周期延拓
3、由于周期性边界影响，束缚态在高能级下的本征态与本征能量计算不准确(除非势场本身就是周期势)
4、Evectors每列为本征值Evalues各元素对应本征态，需要reshape(Evectors[:, j], N, N)还原为2维矩阵
=#
ENV["GKS_ENCODING"] = "utf-8"
using Plots, ToeplitzMatrices, SparseArrays, Arpack, LinearAlgebra
plotlyjs()

# 主函数
function main()
    # 离散设置
    L = 20                        # 求解区域长度
    N = 64                        # 2的整数幂次
    x = L/N*(-N/2:N/2-1)          # 求解区域
    y = copy(x)

    Mesh_X = x' .* ones(N)        # 右为x正向，下为y正向
    Mesh_Y = ones(N)' .* y

    # 二维势函数
    Vfun(x, y) = 1/2*x^2 + 1/2*y^2
    matrix_V = zeros(N, N)
    for i in 1:N
        for j in 1:N
            matrix_V[i, j] = Vfun(Mesh_X[i, j], Mesh_Y[i, j])
        end
    end

    # 构造二阶谱求导矩阵D₂
    h = 2π/N
    vc₀ = [1/2 * (-1)^(j+1) * (csc(j*h/2))^2 for j=1:N-1]
    vc = [-π^2/(3*h^2)-1/6; vc₀]           # D₂ 矩阵的第一列
    D₂ = (2π/L)^2*SymmetricToeplitz(vc)    # 对称常系数矩阵

    # 构建Hamilton矩阵Hami
    m = 1.0     # 电子质量
    Iₙ = sparse(diagm(ones(N)))
    Hami = sparse(-1/(2*m)*(kron(D₂, Iₙ) + kron(Iₙ, D₂)) + diagm(matrix_V[:]))

    # 求解N个本征值E与各点本征函数值ψ
    Evalues, Evectors =@time eigs(Hami; nev=N, which=:SM)   # Arpack的eigs()支持大型稀疏矩阵求特征值；SM为采用特征值由小到大排序
    println("本征能量Eₘₙ = ", Evalues)

    # 本征函数归一化
    Evectors2 = abs2.(Evectors)
    C = sqrt.(1 ./(sum((L/N)^2*Evectors2, dims=1)))  # 归一化系数
    for j=1:N
        Evectors[:, j] = Evectors[:, j]*C[j]
    end

    # 势函数V、前4个本征态ψ绘图
    ψ = zeros(N, N, 4)
    for j in 1:4
        ψ[:, :, j] = reshape(Evectors[:, j], N, N)
    end
    # V(x, y)
    fig1 = plot(x, y, matrix_V,
        xlabel="x (a.u.)", 
        ylabel="y (a.u.)", 
        zlabel="V(x, y)",
        seriestype=:surface,)
    savefig(fig1, "2D_PotentialFuction.png")
    # ψ(x, y)
    fig2 = Any[]
    for j in 1:4
        pic = plot(x, y, ψ[:, :, j],
        xlabel="x (a.u.)", 
        ylabel="y (a.u.)", 
        zlabel="ψ", 
        colorbar=:false, 
        seriestype=:surface, 
        title=string("E = ", 
        round(Evalues[j]; digits = 4), " (a.u.)"))
        push!(fig2, pic)
    end
    fig2 = plot(fig2..., layout=(2, 2))
    savefig(fig2, "2D_EigWaveFuction.png")

    return Evalues, Evectors
end
Evalues, Evectors = main();    # 本征值，本征向量
