from symbol import *

def main_CV():




    env = {}
    env["monte"] = 200

    env["T"] = 1

    env["SimTime"] = 60

    env["step"] = int(env["SimTime"] / env["T"])

    env["q"] = 0.1


    tar = {}
    tar["x_pos"] = 10000

    tar["x_vel"] = -100

    tar["x_acc"] = 0

    T = env["T"]



    F = np.array([[1, T, T**2 / 2],
        [0, 1, T],
        [0, 0, 1]])

    tar["X"] = zeros([3, env["step"]])

    tar["X"][:, 0] = np.array([tar["x_pos"],
        tar["x_vel"], tar["x_acc"]]).T

    G = np.array([T**2 / 2, T, 1]).T 

    q = env["q"]

    for k in range(1, env["step"]):

        tar["X"][:, k] = F @ tar["X"][:, k - 1] +\
            G * q * randn()


    t0 = np.arange(0, env["SimTime"], T) 

    env["T1"] = 3

    step1 = floor(env["SimTime"] / env["T1"])

    m1 = zeros([env["monte"], step1])

    rand1 = randn(env["monte"], step1)

    env["m1_var"] = 5

    a1 = int(env["T1"] / T)


    env["T2"] = 5

    step2 = floor(env["SimTime"] / env["T2"])

    m2 = zeros([env["monte"], step2])

    rand2 = randn(env["monte"], step2)

    env["m2_var"] = 10

    a2 = int(env["T2"] / T)

    RMSm1 = zeros([env["monte"], step1])

    RMSm2 = zeros([env["monte"], step2])

    for k in range(0, step1):

        m1[:, k] = tar["X"][0, a1 * k] +\
            rand1[:, k] * env["m1_var"]

        RMSm1[:, k] = abs(m1[:, k] - tar["X"][0, a1 * k])


    for k in range(0, step2):

        m2[:, k] = tar["X"][0, a2 * k] +\
            rand2[:, k] * env["m2_var"]

        RMSm2[:, k] = abs(m2[:, k] - tar["X"][0, a2 * k])




    subplot(321)

    t1 = np.arange(0, env["SimTime"], env["T1"]) 

    t2 = np.arange(0, env["SimTime"], env["T2"]) 

    plot(t0, tar["X"][0,:], '-k.')

    plot(t1, m1[0, :], 'r^')

    plot(t2, m2[0, :], 'g^')

    title('量测值与真实值')

    subplot(322)

    plot(t1, mean(RMSm1, 0), '-r^')

    plot(t2, mean(RMSm2, 0), '-g^')

    title('量测RMS')


    X1 = zeros([3, step1, env["monte"]])

    H = np.array([[1, 0, 0]])

    R = env["m1_var"] ** 2

    T1 = env["T1"]



    F1 = np.array([[1, T1, T1 ** 2 / 2],
        [0, 1, T1],
        [0, 0, 1]])

    G1 = np.array([[T1 ** 2 / 2], [T1], [1]])

    env["q"] = 10 * env["q"]

    for i in range(0, env["monte"]):

        X1[:, 0, i] = np.array([m1[i, 0], 0, 0]).T

        X1[:, 1, i] = np.array([m1[i, 1], 0, 0]).T

        X1[:, 2, i] = np.array([m1[i, 2],
            (m1[i, 2] - m1[i, 1]) / T1,
            ((m1[i, 2] - m1[i, 1]) / T1 -\
            (m1[i, 1] - m1[i, 0]) / T1) / T1]).T

        P1 = 1e6 * eye(3)

        for k in range(3, step1):

            X_Pre = F1 @ X1[:, k - 1, i]

            Z_Pre = H @ X_Pre

            v = m1[i, k] - Z_Pre

            P_Pre = F1 @ P1 @ F1.T + G1 * q  @ G1.T

            S = H @ P_Pre @ H.T + R

            K = P_Pre @ H.T / S

            X1[:, k, i] = X_Pre + K.reshape([3]) * v

            P1 = P_Pre - K * S @ K.T




    X2 = zeros([3, step2, env["monte"]])

    H = np.array([[1, 0, 0]])

    R = env["m2_var"] ** 2

    T2 = env["T2"]



    F2 = np.array([[1, T2, T2 ** 2 / 2],
        [0, 1, T2],
        [0, 0, 1]])

    G2 = np.array([[T2 ** 2 / 2], [T2], [1]])

    for i in range(0, env["monte"]):

        X2[:, 0, i] = np.array([m2[i, 0], 0, 0]).T

        X2[:, 1, i] = np.array([m2[i, 1], 0, 0])

        X2[:, 2, i] = np.array([m2[i, 2],
            (m2[i, 2] - m2[i, 1]) / T2,
            ((m2[i, 2] - m2[i, 1]) / T2 - \
            (m2[i, 1] - m2[i, 0]) / T2) / T2]).T

        P2 = 1e6 * eye(3)

        for k in range(3, step2):

            X_Pre = F2 @ X2[:, k - 1, i]

            Z_Pre = H @ X_Pre

            v = m2[i, k] - Z_Pre

            P_Pre = F2 @ P2 @ F2.T + G2 * q @ G2.T

            S = H @ P_Pre @ H.T + R

            K = P_Pre @ H.T / S

            X2[:, k, i] = X_Pre + K.reshape([3]) * v

            P2 = P_Pre - K * S @ K.T




    subplot(323)

    plot(t1[2:], X1[0, 2:, 100], '-r*')

    plot(t2[2:], X2[0, 2:, 100], '-g*')

    title('滤波值')

    RMSX1 = zeros([3, step1])

    RMSX2 = zeros([3, step2])

    for idim in range(0, 3):

        for k in range(2, step1):

            RMSX1[idim, k] = mean(abs(X1[idim, k, :] -\
                tar["X"][idim, a1 * k]))


        for k in range(2, step2):

            RMSX2[idim, k] = mean(abs(X2[idim, k, :] -\
                tar["X"][idim, a2 * k]))



    subplot(324)

    plot(t1[2:], RMSX1[0, 2:], '-r*')

    plot(t2[2:], RMSX2[0, 2:], '-g*')

    title('滤波位置')


    a, b, c = X1.shape

    m1 = np.reshape(X1[0, :, :], (b, c))

    m1 = m1.T

    a, b, c = X2.shape

    m2 = np.reshape(X2[0, :, :], (b, c))

    m2 = m2.T

    env["ts"] = 2 * T2

    algorithm = 5


    M1 = zeros([algorithm, env["step"], env["monte"]])

    M2 = zeros([algorithm, env["step"], env["monte"]])

    for i in range(0, env["monte"]):

        for k in range(0, env["step"]):

            if t0[k] <= env["ts"]:

                continue


            index = np.flatnonzero(t1 == t0[k])

            ll = length(index)

            if ll == 1:

                M1[0, k, i] = m1[i, index]

                M1[1, k, i] = m1[i, index]

                M1[2, k, i] = m1[i, index]

                M1[3, k, i] = m1[i, index]

                M1[4, k, i] = X1[0, index, i]

                continue


            index = np.flatnonzero(t1 < t0[k])

            ll = length(index) - 1

            iindex = np.arange(ll - 2, ll + 1)

            M1[0, k, i] = LSFit1time(t1[iindex], m1[i, iindex], t0[k])

            M1[1, k, i] = LSFit2time(t1[iindex], m1[i, iindex], t0[k])

            M1[2, k, i] = Lagrange1time(t1[iindex], m1[i, iindex], t0[k])

            M1[3, k, i] = Lagrange2time(t1[iindex], m1[i, iindex], t0[k])

            M1[4, k, i] = X1[0, ll, i] + X1[1, ll, i] * (t0[k] - t1[ll]) +\
                0.5 * X1[2, ll, i] * (t0[k] - t1[ll]) ** 2


        for k in range(0, env["step"]):

            if t0[k] <= env["ts"]:

                continue


            index = np.flatnonzero(t2 == t0[k])

            ll = length(index)

            if ll == 1:

                M2[0, k, i] = m2[i, index]

                M2[1, k, i] = m2[i, index]

                M2[2, k, i] = m2[i, index]

                M2[3, k, i] = m2[i, index]

                M2[4, k, i] = X2[0, index, i]

                continue


            index = np.flatnonzero(t2 < t0[k])

            ll = length(index) - 1

            iindex = np.arange(ll - 2, ll + 1)

            M2[0, k, i] = LSFit1time(t2[iindex], m2[i, iindex], t0[k])

            M2[1, k, i] = LSFit2time(t2[iindex], m2[i, iindex], t0[k])

            M2[2, k, i] = Lagrange1time(t2[iindex], m2[i, iindex], t0[k])

            M2[3, k, i] = Lagrange2time(t2[iindex], m2[i, iindex], t0[k])

            M2[4, k, i] = X2[0, ll, i] + X2[1, ll, i] * (t0[k] - t2[ll])



    ks = int((env["ts"] + 1) / T) + 1

    subplot(325)

    plot(t0, tar["X"][0, :], '-k.')

    plot(t0[ks:], M1[0, ks:, 0], 'r^')

    plot(t0[ks:], M2[0, ks:, 0], 'g^')

    title('配准值与真实值')


    RMSM1 = zeros([algorithm, env["step"]])

    RMSM2 = zeros([algorithm, env["step"]])

    mark = ['+','o','*','.','d','^']


    plt.figure()









    for i in range(0, algorithm):

        for k in range(ks - 1, env["step"]):

            RMSM1[i, k] = mean(abs(M1[i, k, :] - tar["X"][0, k]))

            RMSM2[i, k] = mean(abs(M2[i, k, :] - tar["X"][0, k]))


        if i == 3:

            continue


        stem(t0[ks:], RMSM1[i, ks:], markerfmt=mark[i],
            linefmt='k:')


    ylim([3, 15])


    xlabel('t\rm(s)') 
    ylabel('RMS(m)')

    legend(['线性最小二乘拟合','二次多项式最小二乘拟合','拉格朗日线性插值','卡尔曼预测'])



    title('时间配准误差')


    plt.show()
    return
if __name__ == '__main__':
    main_CV()
