import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

def MLP(model_size, dtype=tf.float32):
    activation, kernel_initializer, input_dim = model_size[:3]
    output_dim = model_size[-1]
    hidden_units = model_size[3:-1]

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,), dtype=dtype))

    for units in hidden_units:
        model.add(tf.keras.layers.Dense(
                units=units,
                activation=activation,
                kernel_initializer=kernel_initializer,
                dtype=dtype))

    model.add(tf.keras.layers.Dense(
            units=output_dim,
            kernel_initializer=kernel_initializer,
            dtype=dtype))
    return model

def sampling_inside(N_f : int,
                    domain : str,
                    dtype ):
    
    if domain == 'ball':
        points = []
        while len(points) < N_f:
            x, y = np.random.uniform(-1, 1, 2)
            if x**2 + y**2 < 1:
                points.append((x, y))

        points = np.array(points)
        x_inside = points[:, 0]
        y_inside = points[:, 1]
        x_inside_tensor = tf.convert_to_tensor(x_inside.reshape(-1, 1), dtype=dtype)
        y_inside_tensor = tf.convert_to_tensor(y_inside.reshape(-1, 1), dtype=dtype)
        X = tf.concat([x_inside_tensor, y_inside_tensor], axis=1)
    return X, x_inside_tensor, y_inside_tensor
    
def sampling_boundary(N_u : int,
                      domain : str,
                      dtype ):
    
    if domain == 'ball':
        theta = np.random.uniform(0, 2 * np.pi, size=N_u)
        theta = np.sort(theta)
        x_boundary = np.cos(theta)
        y_boundary = np.sin(theta)
        x_boundary_tensor = tf.convert_to_tensor(x_boundary.reshape(-1,1), dtype = dtype)
        y_boundary_tensor = tf.convert_to_tensor(y_boundary.reshape(-1,1), dtype = dtype)
        X = tf.concat([x_boundary_tensor, y_boundary_tensor], axis=1)
    return X, x_boundary_tensor, y_boundary_tensor
    
def sampling_boundary_linspace(N_u : int,
                      domain : str,
                      dtype ):
    
    if domain == 'ball':
        theta = np.linspace(0, 2*np.pi, N_u, endpoint=False)
        x_boundary = np.cos(theta)
        y_boundary = np.sin(theta)
        x_boundary_tensor = tf.convert_to_tensor(x_boundary.reshape(-1,1), dtype = dtype)
        y_boundary_tensor = tf.convert_to_tensor(y_boundary.reshape(-1,1), dtype = dtype)
        X = tf.concat([x_boundary_tensor, y_boundary_tensor], axis=1)
        return X, x_boundary_tensor, y_boundary_tensor
    

def sampling_boundary_randomized(N_u : int,
                                domain : str, 
                                dtype,):
    if domain == 'ball':
        N = int(N_u/2)
        h = tf.constant(1.0, dtype = dtype) / tf.cast(N, dtype)
        tau = tf.random.uniform(shape=(N, 1), minval=0.0, maxval=1.0, dtype=dtype)
        i = tf.cast(tf.range(N), dtype=dtype)  
        i = tf.reshape(i, (N, 1))              
        y1 = i * h + tau * h                   
        y2 = i * h + (1.0 - tau) * h           
        y_zero = tf.zeros((1, 1), dtype = dtype)
        y_points = tf.concat([y_zero, y1, y2], axis=0)
        y_points = tf.sort(y_points, axis=0)
        theta = 2 * np.pi * y_points 
        x = tf.cos(theta)
        y = tf.sin(theta)
        coords = tf.concat([x, y], axis=1)  
    return coords , x, y

def g0(X, g, net):
    return g(X) - net(X)

n_theta = 512

def harmonic_extension(g, k_max=50, n_theta = n_theta, n_theta_rel= 100, n_r=100):
    def g_line (t):
        orig_shape = t.shape
        t = tf.convert_to_tensor(t, dtype=tf.float32)
        x = tf.reshape(tf.cos(t), (-1, 1))
        y = tf.reshape(tf.sin(t), (-1, 1))
        X = tf.concat([x, y], axis=1)
        g_vals = tf.squeeze(g(X), axis=-1)          
        g_vals = g_vals.numpy().reshape(orig_shape) 
        return g_vals
    
    theta = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    g_vals = g_line(theta)

    g_hat = np.fft.rfft(g_vals) / n_theta
    a_0 = g_hat[0].real
    a_k = 2 * g_hat[1:k_max+1].real
    b_k = -2 * g_hat[1:k_max+1].imag

    r = np.linspace(0, 1, n_r)
    R, Theta = np.meshgrid(r, theta)

    U = a_0 * np.ones_like(R)
    for k in range(1, k_max+1):
        U += R**k * (a_k[k-1] * np.cos(k * Theta) + b_k[k-1] * np.sin(k * Theta))

    theta_100 = np.linspace(0, 2*np.pi, n_theta_rel, endpoint=False)
    R_100, Theta_100 = np.meshgrid(r, theta_100)
    U_100 = a_0 * np.ones_like(R_100)
    U_x = np.zeros_like(R_100)
    U_y = np.zeros_like(R_100)
    for k in range(1, k_max+1):
        U_100 += (R_100)**k * (a_k[k-1] * np.cos(k * Theta_100) + b_k[k-1] * np.sin(k * Theta_100))    
        cos_km1 = np.cos((k-1) * Theta_100)
        sin_km1 = np.sin((k-1) * Theta_100)
        U_x += k * (R_100)**(k-1) * (a_k[k-1] * cos_km1 + b_k[k-1] * sin_km1)
        U_y += k * (R_100)**(k-1) * (-a_k[k-1] * sin_km1 + b_k[k-1] * cos_km1)

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    U = np.vstack([U, U[0:1, :]])
    X = np.vstack([X, X[0:1, :]])
    Y = np.vstack([Y, Y[0:1, :]])

    X_100 = R_100 * np.cos(Theta_100)
    Y_100 = R_100 * np.sin(Theta_100)

    U_100 = np.vstack([U_100, U_100[0:1, :]])
    U_x = np.vstack([U_x, U_x[0:1, :]])
    U_y = np.vstack([U_y, U_y[0:1, :]])
    X_100 = np.vstack([X_100, X_100[0:1, :]])
    Y_100 = np.vstack([Y_100, Y_100[0:1, :]])
    return U_100, X_100, Y_100, U_x, U_y, U, X, Y #_100 is for calculation of relative error

@tf.function
def Loss(IN,BD, net, f, g, rate_f, rate_u, rate_g, domain:str):
    if domain == 'ball':
        x_inside_tensor = IN[:,0:1]
        y_inside_tensor = IN[:,1:2]

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_inside_tensor)
            tape.watch(y_inside_tensor)

            u_inside_pred = net(tf.concat([x_inside_tensor, y_inside_tensor], axis=1))
            u_x = tape.gradient(u_inside_pred, x_inside_tensor)
            u_y = tape.gradient(u_inside_pred, y_inside_tensor)
            u_xx = tape.gradient(u_x, x_inside_tensor)
            u_yy = tape.gradient(u_y, y_inside_tensor)
            PDE_u = u_xx + u_yy
            
            MSE_f = tf.reduce_mean(tf.square(PDE_u - f(IN)))
            MSE_u = tf.reduce_mean(tf.square(g0(BD, g, net)))
            q = g0(BD, g, net)
            q_shifted = tf.concat([q[1:], q[:1]], axis=0)
            q_shifted2 = tf.concat([q_shifted[1:], q_shifted[:1]], axis=0)
            MSE_g = tf.reduce_sum((q-q_shifted)**2 + (q-q_shifted2)**2)
            
            MSE_TRPINN = rate_f * MSE_f + rate_u * MSE_u + rate_g * MSE_g
            MSE_PINN = rate_f * MSE_f + rate_u * MSE_u
        return MSE_TRPINN , MSE_f, MSE_u, MSE_g, MSE_PINN
    
def rel_er(u,v,ux,uy,vx,vy):
    L2_n = tf.sqrt(tf.reduce_mean(tf.square(u-v)))
    L2D1_n = tf.sqrt(tf.reduce_mean(tf.square(ux-vx)))
    L2D2_n = tf.sqrt(tf.reduce_mean(tf.square(uy-vy)))
    L2_d = tf.sqrt(tf.reduce_mean(tf.square(v)))
    L2D1_d = tf.sqrt(tf.reduce_mean(tf.square(vx)))
    L2D2_d = tf.sqrt(tf.reduce_mean(tf.square(vy)))

    rel_H1 = (L2_n + L2D1_n + L2D2_n)/(L2_d + L2D1_d + L2D2_d)
    rel_L2 = L2_n/L2_d
    H1 = L2_n + L2D1_n + L2D2_n
    return rel_H1, rel_L2, H1


@tf.function
def rel_er_H1_INSIDE(IN_test, net, domain:str, Solution = None):
    if domain == 'ball':
        net_in = net(IN_test)
        if callable(Solution):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(IN_test)
                net_in = net(IN_test)
                Sol = Solution(IN_test)
            du = tape.gradient(net_in, IN_test)
            ds = tape.gradient(Sol, IN_test)
            ux = du[:, 0:1]
            uy = du[:, 1:2]
            Solx = ds[:, 0:1]
            Soly = ds[:, 1:2]
            Inside_error = rel_er(net_in, Sol, ux, uy, Solx, Soly)
        elif Solution == None:
            Inside_error = [tf.constant(np.nan, dtype=IN_test.dtype), None]
        else: # FFT harmonic extension solution
            Sol = tf.reshape(tf.cast(Solution[1][0], dtype = IN_test.dtype), [-1,1])
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(IN_test)
                net_in = net(IN_test)
            du = tape.gradient(net_in, IN_test)
            ux = du[:, 0:1]
            uy = du[:, 1:2]
            Solx = tf.reshape(tf.cast(Solution[1][3], IN_test.dtype), [-1, 1])
            Soly = tf.reshape(tf.cast(Solution[1][4], IN_test.dtype), [-1, 1])
            Inside_error = rel_er(net_in, Sol, ux, uy, Solx, Soly)
    return Inside_error[0]

def make_weight_matrix(N, dtype=tf.float32):
    I = tf.range(N)[:, None]
    J = tf.range(N)[None, :]
    W = tf.ones((N, N), dtype=dtype)
    W = tf.where((I == 0) & (J == N-1), tf.cast(0.5, dtype), W)
    W = tf.where(((I == 0) | (J == N-1)) & ~((I == 0) & (J == N-1)),
                 tf.cast(0.75, dtype), W)
    W = tf.where(I > J, tf.zeros_like(W), W)
    return W

def num(X):
    X = tf.convert_to_tensor(tf.reshape(X, [-1]))  
    M = X[:, None] - X[None, :]                    
    M = tf.linalg.set_diag(M, tf.zeros_like(X))    
    return M

def den(X):
    D = tf.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    NN = tf.shape(D)[0]
    D = tf.linalg.set_diag(D, tf.ones([NN], dtype=D.dtype))
    return D

def Semi_er(u, v, BD):
    N_b = tf.shape(v)[0]
    W = make_weight_matrix(N_b, dtype=v.dtype)
    NU = (num(u-v))**2
    DE = (den(BD))**2
    seminorm = tf.sqrt((2*(2*np.pi)**2)*tf.reduce_sum(W*(NU/DE)))
    return seminorm

def rel_er_halfbd(u,v,BD):
    L2_n = tf.sqrt(2*np.pi*tf.reduce_mean(tf.square(u-v)))
    semi_n = Semi_er(u, v, BD)
    L2_d = tf.sqrt(2*np.pi*tf.reduce_mean(tf.square(v)))
    semi_d = Semi_er(0, v, BD)

    rel_H = (L2_n + semi_n)/(L2_d + semi_d)
    rel_L2 = L2_n/L2_d
    return rel_H, rel_L2

@tf.function
def relative_error(IN_test, BD_test, net, g, domain:str, Solution = None):
    if domain == 'ball':
        Boundary_error = rel_er_halfbd(net(BD_test), g(BD_test), BD_test)
        if callable(Solution):
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(IN_test)
                net_in = net(IN_test)
                Sol = Solution(IN_test)
            du = tape.gradient(net_in, IN_test)
            ds = tape.gradient(Sol, IN_test)
            ux = du[:, 0:1]
            uy = du[:, 1:2]
            Solx = ds[:, 0:1]
            Soly = ds[:, 1:2]
            Inside_error = rel_er(net_in, Sol, ux, uy, Solx, Soly)
        elif Solution == None:
            Inside_error = [tf.constant(np.nan, dtype=IN_test.dtype), tf.constant(np.nan, dtype=IN_test.dtype), tf.constant(np.nan, dtype=IN_test.dtype), tf.constant(np.nan, dtype=IN_test.dtype)]
        else: # FFT harmonic extension solution
            Sol = tf.reshape(tf.cast(Solution[1][0], dtype = IN_test.dtype), [-1,1])
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(IN_test)
                net_in = net(IN_test)
            du = tape.gradient(net_in, IN_test)
            ux = du[:, 0:1]
            uy = du[:, 1:2]
            Solx = tf.reshape(tf.cast(Solution[1][3], IN_test.dtype), [-1, 1])
            Soly = tf.reshape(tf.cast(Solution[1][4], IN_test.dtype), [-1, 1])
            Inside_error = rel_er(net_in, Sol, ux, uy, Solx, Soly)
    return Inside_error[0], Inside_error[1], Boundary_error[0], Boundary_error[1]

def Visualization(Networks,
                  name, 
                  domain:str, 
                  Vis_show:str,
                  Save_dir:str,
                  rel_errors,
                  Errors_record = None,
                  Solution = None):
    
    ex_values = []
    ey_values = []
    ex_values = np.arange(len(Errors_record))
    ey_values = np.array(Errors_record)

    def NETWORK(x, y):
        inputs = np.stack((x, y), axis=-1).reshape(-1, 2)
        return Networks.predict(inputs).reshape(x.shape)
    
    if callable(Solution):
        def SOLUTION(x, y):
            X = np.stack((x, y), axis=-1).reshape(-1, 2)
            return Solution(X)

    if domain == 'ball':

        theta = np.linspace(0, 2 * np.pi, n_theta + 1)
        r = np.linspace(0, 1, 100)

        theta = theta.astype(np.float32)
        r = r.astype(np.float32)

        R, Theta = np.meshgrid(r, theta)
        X, Y = R * np.cos(Theta), R * np.sin(Theta)

        X_bd = np.cos(theta)
        Y_bd = np.sin(theta)


        Z = NETWORK(X, Y)
        Z_bd = NETWORK(X_bd, Y_bd).reshape(-1,1)

        title = 'Solution'

        if callable(Solution):
            Sol = np.reshape(SOLUTION(X, Y), X.shape)
            Sol_bd = np.reshape(SOLUTION(X_bd, Y_bd), X_bd.shape)
        elif Solution == None:
            Sol = Z*0
            Sol_bd = Z_bd* 0
            title = 'Unkown solution'
        else:
            Sol = Solution[1][5] 
            Sol_bd = Solution[0](np.stack([X_bd, Y_bd], axis=-1))

    levels = 1000

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1,
        figsize=(15, 10)
    )
    contour = ax1.contourf(X, Y, Sol, levels=levels, cmap='viridis', alpha=1.0 )
    ax1.set_title('Inside')
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_aspect('equal')
    fig.colorbar(contour, ax=ax1, shrink=1.0, pad=0.01, aspect=20)

    ax2.plot(theta, Sol_bd, color='red', alpha=1.0 )
    ax2.set_title('Boundary')
    ax2.set_xlabel(r'$\theta$')
    ax2.legend()

    if Vis_show == 'show':
        plt.show()
    elif Vis_show == 'save':
        filename = f"Solution.png"  
        fullpath = os.path.join(Save_dir, filename)
        fig.savefig(fullpath, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(6, 1, figsize=(15, 28.0),
                            gridspec_kw={'height_ratios': [1,1,1,1,1, 0.5]},
                            constrained_layout=True
                            )

    contour1 = axes[0].contourf(X, Y, np.ma.array(Z), levels=levels, cmap='viridis')
    axes[0].set_title('Inside')
    axes[0].set_xlim([-1, 1])
    axes[0].set_ylim([-1, 1])
    axes[0].set_aspect('equal')
    fig.colorbar(contour1, ax = axes[0], shrink=1.0, pad=0.01, aspect=20)

    contour2 = axes[1].contourf(X, Y, abs(Sol - np.ma.array(Z)), levels=levels, cmap='magma')
    axes[1].set_title('Absolute')
    axes[1].set_xlim([-1, 1])
    axes[1].set_ylim([-1, 1])
    axes[1].set_aspect('equal')
    fig.colorbar(contour2, ax = axes[1], shrink=1.0, pad=0.01, aspect=20)

    axes[2].plot(theta, Z_bd, color='blue')
    axes[2].plot(theta, Sol_bd, color='red', linestyle='--')
    axes[2].set_title('Boundary')
    axes[2].set_xlabel(r'$\theta$')
    axes[2].set_ylabel('Function Value')
    axes[2].legend()

    axes[3].plot(theta, np.abs(Sol_bd-Z_bd))
    axes[3].set_title('Absolute')
    axes[3].set_xlabel(r'$\theta$')
    axes[3].set_ylabel('Function Value')
    axes[3].legend()

    axes[4].semilogy(ex_values, ey_values, linestyle='-')
    axes[4].set_title(f'Relative error')
    axes[4].set_xlabel('Iteration')
    axes[4].set_ylabel('Error')
    axes[4].legend()

    axes[5].axis('off')

    data1 = [
        ["Relative errors"],
        [r"$H^1$ inside",      rel_errors[0]],
        [r"$L^2$ inside",       rel_errors[1]],
        [r"$H^{1/2}$ boundary",       rel_errors[2]],
        [r"$L^2$ boundary",       rel_errors[3]],
    ]
    table_text1 = "\n".join(f"{row[0]} : {row[1]}" if len(row) > 1 else row[0]
    for row in data1)
    axes[5].text(0.01, 0.5, table_text1, va='center', fontsize=20)

    fig.suptitle(f"{name}.", fontsize=20, y=0.98)
    if Vis_show == 'show':
        plt.show()
    elif Vis_show == 'save':
        filename = f"{name}, visualization.png"  
        fullpath = os.path.join(Save_dir, filename)
        fig.savefig(fullpath, pad_inches=1.0, dpi=300)
        plt.close(fig)

def tf_theta(x,y):
    t = tf.atan2(y,x)
    result = tf.where(t>0, t, t + 2*np.pi)
    return result