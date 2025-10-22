import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from trpinn_functions import (
    MLP,
    sampling_inside,
    sampling_boundary,
    sampling_boundary_randomized,
    Loss,
    rel_er_H1_INSIDE,
    Visualization,
    relative_error,
    harmonic_extension,
    tf_theta,
)
from pathlib import Path


####### Hyperparameters ######
dtype = tf.float32
Vis_show = 'save' #'show', 'save'
Save_dir = str(Path(__file__).resolve().parent / "results")  # ./results 폴더
Path(Save_dir).mkdir(parents=True, exist_ok=True)

IR = 10000 # Print training status every IR iterations

name = 'TRPINN' # file name for saving
boundary_loss = 'half' # 'half' is for TRPINN and 'L2' is for Vanilla PINN  
sampling_num_in = 10000 # number of sampled points in the domain interior
sampling_num_bd = 200 # number of sampled points on the boundary

iteration_adam = 50000 # train in two stages: first run Adam for these iterations
iteration_LBFGS = 50000  # then switch to L-BFGS for these iterations (Adam → L-BFGS)

model_size = ['tanh', 'glorot_normal', 2, 128, 128, 128, 1] #[activation, initialization, input, layer,.., layer, output]

loss_weights = [1, 5, 5] # [interior, L_2 boundary, semi-norm boundary]


###### Conditions for the Laplace equation ######
domain = 'ball' # 'ball'

def f(X): # forcing term
    return tf.zeros_like(X[:, :1])

def g(X): # bounary condition
    x = X[:,0:1]
    y = X[:,1:2]
    theta = tf_theta(x,y)
    result = tf.sin(20*theta)
    return result

Sol_FFT = harmonic_extension(g) # Compute the numerical solution via FFT-based harmonic extension of g
#Sol = [g, Sol_FFT] # for evaluation of relative errors and visualization 

#Sol = None # if the exact solution is unknown
#def Sol(X): # if we know the exact solution (callable)
#    x = X[:,0:1]
#    y = X[:,1:2]
#    theta = tf_theta(x,y)
#    result = tf.sin(10*theta)
#    return result


###### Surrogate model ######
Neural_Net = MLP(model_size, dtype = dtype) # Multi-layer Perceptron


###### Sample collocation points for training: ######
Inside = sampling_inside(sampling_num_in, domain, dtype)[0]
Boundary_ran = sampling_boundary_randomized(sampling_num_bd, domain, dtype)[0]


###### Points for relative error calculation ######
if callable(Sol) or Sol == None:
    IN_test = sampling_inside(sampling_num_in, domain, dtype)[0]
else: # Test points where the numerical solution (FFT) is defined
    x_inside ,y_inside = tf.convert_to_tensor(Sol_FFT[1], dtype = dtype), tf.convert_to_tensor(Sol_FFT[2], dtype = dtype)
    IN_test = tf.concat([tf.reshape(x_inside, [-1,1]), tf.reshape(y_inside, [-1,1])], axis =1) 
BD_test = sampling_boundary(200, domain, dtype)[0]


###### Optimization: Adam + L-BFGS ######
optim = tf.keras.optimizers.Adam(learning_rate=0.001)
Errors_nets   = []
Loss_nets     = []

if iteration_adam > 0:
    for itr in range(iteration_adam):
        with tf.GradientTape() as tape:
            T = Loss(Inside, Boundary_ran, Neural_Net, f, g, loss_weights[0], loss_weights[1], loss_weights[2],  domain)
            if boundary_loss == 'half':
                train_loss = T[0]
            elif boundary_loss == 'L2':
                train_loss = T[4]
            Loss_nets.append(train_loss)

            grad_w = tape.gradient(train_loss, Neural_Net.trainable_variables)
            optim.apply_gradients(zip(grad_w, Neural_Net.trainable_variables))

        E = rel_er_H1_INSIDE(IN_test, Neural_Net, domain = domain, Solution = Sol)
        Errors_nets.append(E.numpy()) 
        if itr % IR == 0 or itr == iteration_adam -1:
                print('--------------------------------------------------------------------------------------------')
                print(f'[{itr:>4}], Adam, {name}')
                print("Loss | MSE={:.6e} | MSE_f={:.6e} | MSE_g={:.6e} | semi_g={:.6e}".format(T[0].numpy().item(), T[1].numpy().item(), T[2].numpy().item(), T[3].numpy().item()))
                print("Relative error | H^1 inside={:.6e}".format(E.numpy().item()))
                print('--------------------------------------------------------------------------------------------')

if iteration_LBFGS > 0:
    def get_weights():
        weights = []
        for layer in Neural_Net.layers:
            for var in layer.variables:
                weights.append(var.numpy().flatten())
        return np.concatenate(weights)

    def set_weights(flat_weights):
        start = 0
        for layer in Neural_Net.layers:
            shapes = [var.shape for var in layer.variables]
            for var, shape in zip(layer.variables, shapes):
                size = np.prod(shape)
                new_values = flat_weights[start:start + size].reshape(shape)
                var.assign(new_values)
                start += size

    def loss_and_grads(weights):
        set_weights(weights)
        with tf.GradientTape() as tape:
            loss = Loss(Inside, Boundary_ran, Neural_Net, f, g, loss_weights[0], loss_weights[1], loss_weights[2],  domain)
            if boundary_loss == 'half':
                train_loss = loss[0]
            elif boundary_loss == 'L2':
                train_loss = loss[4]
            Loss_nets.append(train_loss)
        grads = tape.gradient(train_loss, Neural_Net.trainable_variables)
        flat_grads = np.concatenate([a.numpy().flatten() for a in grads])
        return train_loss.numpy(), flat_grads, loss

    def callback(weights):
        current_iteration = len(Errors_nets)
        E = rel_er_H1_INSIDE(IN_test, Neural_Net, domain = domain, Solution = Sol)
        Errors_nets.append(E.numpy())

        if current_iteration % IR == 0 :
            current_loss, _, Loss  = loss_and_grads(weights)
            print('--------------------------------------------------------------------------------------------')
            print(f'[{current_iteration:>4}], L-BFGS, {name}')
            print("Loss | MSE={:.6e} | MSE_f={:.6e} | MSE_g={:.6e} | semi_g={:.6e}".format(current_loss.item(), Loss[1].numpy().item(), Loss[2].numpy().item(), Loss[3].numpy().item()))
            print("Relative error | H^1 inside={:.6e}".format(E.numpy().item()))
            print('--------------------------------------------------------------------------------------------')

    initial_weights = get_weights()
    result = minimize(fun=loss_and_grads,
                        x0=initial_weights,
                        jac=True,
                        method='L-BFGS-B',
                        callback=callback,  
                        tol = 1e-1000000,
                        options={'maxiter': iteration_LBFGS, 'disp': False})
    set_weights(result.x)


###### Report ######
rel_errors = relative_error(IN_test,BD_test, Neural_Net, g, domain = domain, Solution = Sol)

print('--------------------------------------------------------------------------------------------')
print(f'Relative errors of {name}')
print("H^1_inside={:.6e} | L^2_inside={:.6e} | Half_boundary={:.6e} | L^2_boundary={:.6e}".format(rel_errors[0].numpy().item(), rel_errors[1].numpy().item(), rel_errors[2].numpy().item(), rel_errors[3].numpy().item()))
print('--------------------------------------------------------------------------------------------')

weights_path = f"{Save_dir}/{name}.weights.h5"  
Neural_Net.save_weights(weights_path) # save the neural network's weights to disk

out_dir = Path(Save_dir)
Error_record_path = out_dir / f"Error_record_{name}.npy"
Loss_record_path = out_dir / f"Loss_record_{name}.npy"
np.save(Error_record_path, np.array(Errors_nets))
np.save(Loss_record_path, np.array(Loss_nets))

# Vis provides visualizations of the exact solution, predictions in the interior and on the boundary, and the corresponding relative errors.
Vis = Visualization(Neural_Net,
                    name, 
                    domain = domain,
                    Vis_show = Vis_show,
                    Save_dir = Save_dir,
                    rel_errors = rel_errors,
                    Errors_record = Errors_nets,
                    Solution = Sol)
