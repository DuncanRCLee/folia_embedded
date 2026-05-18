# %%
import pyeskf
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sqrt
g  = 9.80665;



# %%
params = pyeskf.Params();
# params.sigma_an = 0.05;
# params.sigma_wn = 0.002;
# params.sigma_aw = 0.0005;
# params.sigma_ww = 0.001;

Fs = 100;

params.sigma_an = 70e-6 * g * sqrt(Fs);
params.sigma_wn = radians(0.003) * sqrt(Fs);
params.sigma_aw = (40e-6 * g) / sqrt(3600);
params.sigma_ww = radians(6) / sqrt(3600);


data = np.loadtxt('test_MTi_stand_1.csv', delimiter=',', skiprows=1)
data_walk = np.loadtxt('test_MTi_walk_1.csv', delimiter=',', skiprows=1)
for k in range(0,1):
    if k == 0:
        dt_list = data[:, 1]
        dt_list = dt_list[:, None]
        a_m_list = data[:, 2:5]
        w_m_list = data[:, 5:8]
        n = len(dt_list)
    else:
        dt_list = data_walk[:, 1]
        dt_list = dt_list[:, None]
        a_m_list = data_walk[:, 2:5]
        w_m_list = data_walk[:, 5:8]
        n = len(dt_list)

    datatype = "Standing Data" if k == 0 else "Walking Data"

    print(f"avg dt for {datatype} is {np.average(dt_list) * 1000}ms")

    ax = plt.figure().gca();
    ax.set_title(datatype)
    ax.plot(a_m_list[:,0], label="raw accel x");
    ax.plot(a_m_list[:,1], label="raw accel y");
    ax.plot(a_m_list[:,2], label="raw accel z");
    plt.legend();

    cumsum_v = np.cumsum(a_m_list * dt_list, axis=0)
    cumsum_p = np.cumsum(cumsum_v * dt_list, axis=0)

    ax = plt.figure().gca()
    ax.set_title("x path vs time")


    ax2 = plt.figure().add_subplot(projection='3d')
    ax2.set_title("world path")

    ax3 = plt.figure().gca()
    ax3.set_title("accel bias vs time")



    #ax.plot(cumsum_p[:,0], label='cumsum p')


    for j in range(0,10):
        f = pyeskf.ESKF(params);

        log_p = np.zeros([n,3])
        log_v = np.zeros([n,3])
        log_a_b = np.zeros([n,3])

        for i in range(0,n):

            a_m = a_m_list[i,:]
            w_m = w_m_list[i,:]
            dt = dt_list[i]

            f.predict(a_m, w_m, dt)

            if j > 0:
                y = np.zeros([3,1])
                Hx = np.zeros([3,15])
                Hx[:,3:6] = np.eye(3)
                Rvv = (0.04 / j)**2 * np.eye(3)
                hFun = lambda f: f.v

                f.correct(y,Hx,Rvv,hFun)


            log_p[i] = f.p
            log_v[i] = f.v
            log_a_b[i] = f.a_b


        ax3.plot(log_a_b[:,0], label=f'learned ab for {j}')

        if j == 0:
            ax.plot(log_p[:,0], label='X position, no correction')
            #ax2.plot(log_p[:,0], log_p[:,1], log_p[:,2], label='Xvec, no correction')
        if j > 0:
            ax.plot(log_p[:,0], label=f'X position, correction {j}')
            ax2.plot(log_p[:,0], log_p[:,1], log_p[:,2], label=f'Xvec, correction {j}')


    plt.legend()



# plt.figure()
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot(badLogP[:,0],badLogP[:,1],badLogP[:,2], label="bad position over time")
# ax.plot(goodLogP[:,0],goodLogP[:,1],goodLogP[:,2], label="good position over time")
# ax.legend()



# # %%
# # Plot the trajectory
# ax.plot(log_p[:,1], label='Y')
# ax.plot(log_p[:,2], label='Z')
# plt.legend()
# plt.title('Position Trajectory')

# plt.figure()
# ax = plt.gca()
# ax.plot(log_v[:,0], label='vX')
# ax.plot(log_v[:,1], label='vY')
# ax.plot(log_v[:,2], label='vZ')
# plt.legend()
# plt.title('Velocity Trajectory')


# plt.subplot(2, 1, 2)
# ax = plt.gca()
# ax.plot(log_p[:,0], 'r-', log_v[:,0], 'b-')
# ax.plot(log_p[:,1], 'g-', log_v[:,1], 'm-')
# ax.plot(log_p[:,2],)
# ... (rest of the code would follow here, but was not provided)

# %%

print("Position:", f.p)
print("Velocity:", f.v)
#print("Orientation (quaternion):", f.q)
