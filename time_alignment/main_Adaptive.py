from symbol import *

def main_Adaptive():




    env = {}
    env["monte"] = 200

    env["T"] = 1

    env["SimTime"] = 72

    env["step"] = int(env["SimTime"] / env["T"])


    tar = {}
    tar["x_pos"] = 10000

    tar["x_vel"] = -10

    tar["x_acc"] = 0

    tar["x_add"] = 5

    tar["Tchange"] = np.array([60,65,100,125])

    T = env["T"]



    F = np.array([[1, T, T**2 / 2],
        [0, 1, T],
        [0, 0, 1]])

    X = zeros([3, env["step"], env["monte"]])

    G = np.array([T**2 / 2, T, 1]).T

    q = 0.1

    t0 = np.arange(0, env["SimTime"], env["T"]) 



    T1 = 3 * env["T"]

    step1 = floor(env["SimTime"] / T1)

    m1 = zeros([env["monte"], step1])

    rand1 = randn(env["monte"], step1)

    m1_var = 5

    a1 = int(T1 / env["T"])

    T2 = 5 * env["T"]

    step2 = floor(env["SimTime"] / T2)

    m2 = zeros([env["monte"], step2])

    rand2 = randn(env["monte"], step2)

    m2_var = 10

    a2 = int(T2 / env["T"])

    RMSm1 = zeros([env["monte"], step1])

    RMSm2 = zeros([env["monte"], step2])

    for i_monte in range(0, env["monte"]):

        X[:, 0, i_monte] = np.array([
            tar["x_pos"],tar["x_vel"],tar["x_acc"]])

        for k in range(1, env["step"]):

            X[:, k, i_monte] = F @ X[:, k - 1, i_monte] +\
                G * q * randn()

            if k == tar["Tchange"][0]:

                X[2, k, i_monte] = 5


            if k >= tar["Tchange"][1]:

                X[2, k, i_monte] = X[2, k, i_monte] + tar["x_add"]



        for k in range(0, step1):

            m1[i_monte, k] = X[0, a1 * k, i_monte] +\
                rand1[i_monte, k] * m1_var

            RMSm1[i_monte, k] = abs(m1[i_monte, k] - X[0, a1 * k, i_monte])


        for k in range(0, step2):

            m2[i_monte, k] = X[0, a2 * k, i_monte] +\
                rand2[i_monte, k] * m2_var

            RMSm2[i_monte, k] = abs(m2[i_monte, k] - X[0, a2 * k, i_monte])






    subplot(321)

    t1 = np.arange(0, env["SimTime"], T1)

    t2 = np.arange(0, env["SimTime"] - T2, T2)



    plot(t0, X[0, :, 0], '-k.')

    plot(t1, m1[0, :], 'r^')

    plot(t2, m2[0, :], 'g^')

    title('量测值与真实值')

    subplot(322)

    plot(t1, mean(RMSm1, 0), '-r^')

    plot(t2, mean(RMSm2, 0), '-g^')

    title('量测RMS')



    X1 = zeros([3, step1, env["monte"]])

    H = np.array([[1, 0, 0]])

    R = m1_var ** 2



    F1 = np.array([[1, T1, T1 ** 2 / 2],
        [0, 1, T1],
        [0, 0, 1]])

    G1 = np.array([[T1 ** 2 / 2], [T1], [1]])

    q = 10 * q

    for i in range(0, env["monte"]):

        X1[:, 0, i] = np.array([m1[i, 0], tar["x_vel"], tar["x_acc"]])

        X1[:, 1, i] = np.array([m1[i, 1],
        (m1[i, 1] - m1[i, 0]) / T1, tar["x_acc"]])


        X1[:, 2, i] = np.array([m1[i, 2],
            (m1[i, 2] - m1[i, 1]) / T1,
            ((m1[i, 2] - m1[i, 1]) / T1 -\
            (m1[i, 1] - m1[i, 0]) / T1) / T1]).T




        P1 = np.array([[m1_var, m1_var / T1, m1_var / (T1 ** 2)],
            [m1_var / T1, 2 * m1_var / (T1 ** 2), 3 * m1_var / (T1 ** 3)],
            [m1_var / (T1 ** 2), 3 * m1_var / (T1 ** 3),
            6 * m1_var / (T1 ** 4)]])

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

    R = m2_var ** 2



    F2 = np.array([[1, T2, T2 ** 2 / 2],
        [0, 1, T2],
        [0, 0, 1]])

    G2 = np.array([[T2 ** 2 / 2], [T2], [1]])

    for i in range(0, env["monte"]):

        X2[:, 0, i] = np.array([m2[i, 0], 0, 0]).T

        X2[:, 1, i] = np.array([m2[i, 1], 0, 0]).T

        X2[:, 2, i] = np.array([m2[i, 2],
            (m2[i, 2] - m2[i, 1]) / T2,
            ((m2[i, 2] - m2[i, 1]) / T2 - \
            (m2[i, 1] - m2[i, 0]) / T2) / T2]).T

        P2 = 1e4 * eye(3)

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

    dd = mean(X1[2, :, :], axis=1)

    plot(t1[2:], dd[2:], '-r*')

    title('滤波值')

    RMSX1 = zeros([3, step1])

    RMSX2 = zeros([3, step2])

    for idim in range(0, 3):

        for k in range(2, step1):

            RMSX1[idim, k] = mean(abs(X1[idim, k, :] -\
                X[idim, a1 * k]))


        for k in range(2, step2):

            RMSX2[idim, k] = mean(abs(X2[idim, k, :] -\
                X[idim, a2 * k]))



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

    ts = 2 * T2

    algorithm = 6

    M1 = zeros([algorithm, env["step"], env["monte"]])

    M2 = zeros([algorithm, env["step"], env["monte"]])

    flag = zeros([env["step"], env["monte"]])

    vava = zeros([env["step"], env["monte"]])

    for i in range(0, env["monte"]):

        for k in range(0, env["step"]):

            if k == env["step"] - 1:

                a = 1 


            if t0[k] <= ts:
                continue



            index = np.flatnonzero(t1 == t0[k])

            ll = length(index)

            if ll == 1:

                M1[0, k, i] = m1[i, index]

                M1[1, k, i] = m1[i, index]

                M1[2, k, i] = m1[i, index]

                M1[3, k, i] = m1[i, index]

                M1[4, k, i] = X1[0, index, i]

                M1[5, k, i] = X1[0, index, i]

                flag[k, i] = 1

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
        

            a11 = (X1[1, ll - 1, i] - X1[1, ll - 2, i]) / T1

            a22 = (X1[1, ll, i] - X1[1, ll - 1, i]) / T1

            vava[k, i] = abs(a22)

            env["h1"] = 3

            env["h2"] = 0.7

            if abs(a11 - a22) / T1 > env["h2"]:

                flag[k, i] = 4

                M1[5, k, i] = LSFit2time(t1[iindex], m1[i, iindex], t0[k])

            else:

                if abs(a22) > env["h1"]:

                    flag[k, i] = 3

                    M1[5, k, i] = X1[0, ll, i] + X1[1, ll, i] * (t0[k] - t1[ll]) +\
                        0.5 * X1[2, ll, i] * (t0[k] - t1[ll]) ** 2

                else:

                    flag[k, i] = 2

                    M1[5, k, i] = LSFit1time(t1[iindex], m1[i, iindex], t0[k])






    ks = int((ts + 1) / env["T"]) + 1

    subplot(325)

    plot(t0, X[0, :, 0], '-k.')

    plot(t0[ks:], M1[0, ks:, 0], 'r^')


    title('配准值与真实值')


    RMSM1 = zeros([algorithm, env["step"]])

    RMSM2 = zeros([algorithm, env["step"]])

    error = zeros(algorithm)

    mark = ['+','o','*','.','d','^']


    plt.figure()










    for i in range(0, algorithm):

        for k in range(ks - 1, env["step"]):

            RMSM1[i, k] = mean(abs(M1[i, k, :] - X[0, k, :]))

            RMSM2[i, k] = mean(abs(M2[i, k, :] - X[0, k, :]))


        if i == 3 or i == 2:

            continue



        stem(t0[ks:-1], RMSM1[i, ks:-1], markerfmt=mark[i],
            linefmt='k:')

        error[i] = mean(RMSM1[i, ks:])



    xlabel('t\rm(s)')
    ylabel('RMS(m)')

    legend(['线性最小二乘拟合','二次多项式最小二乘拟合','卡尔曼预测','自适应配准'])




    ylim([0, 160])

    plt.show()
    return
if __name__ == '__main__':
    main_Adaptive()
