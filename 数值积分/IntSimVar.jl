# 采用变步长Simpson法解单变量实数定积分
# 输入被积函数f，积分上限a，下限b，误差限eps(输入函数f必须支持点运算操作)
# 输出积分值S2N，残差epsm, 划分次数m
function IntSimVar(f::Function, a::Float64, b::Float64, eps::Float64)
    N = 100          # 初始分段数
    h = (b - a)/N    # 初始步长
    TN = 0.0 ; T2N = 0.0 ;
    SN = 0.0 ; S2N = 0.0 ;
    e1 = 0.0 ; e2 = 0.0 ; epsm = 0.0 ;
    m = 1            # 迭代次数

    # 计算初始复化梯形求积TN
    X = Array{Float64}(a : h : b)     # 得到X的范围向量，相当于linspace()
    TN = h*(1/2*f(a) + 1/2*f(b) + sum(f(X[2:N])))

    # 二分步长迭代
    while true
        k1 = Array(0 : N-1)      # 求和序列
        k2 = Array(0 : 2N-1)
        T2N = TN/2 + h/2*sum(f(a.+(k1.+1/2).*h))  # 步长减半复化梯形

        # 由TN、T2N构造复化Simpson求积、减半步长Simpson求积
        SN = T2N + 1/3*(T2N - TN)
        S2N = T2N/3 + h/3*sum(f(a.+(k2.+1/2).*h/2))

        # 计算残差，跳出循环
        e1 = abs(S2N - SN)
        e2 = abs((S2N - SN)/S2N)
        if abs(S2N)<=1 && e1<eps
            epsm = e1
            break
        elseif abs(S2N)>1 && e2<eps
            epsm = e2
            break
        elseif m > 12
            epsm = "error"
            print("Error: 超出最大分化次数")
            break
        end

        # 重构步长h、分段数N、复化梯形求积TN
        N = 2N
        h = h/2
        TN = T2N
        m += 1
    end

    return S2N, epsm, m
end
