
## Configurable Parameters:


### [DNN.size]

ngru (Number of GRU layers)

gru_scale_s (Scaling factor for S)

gru_scale_k (Scaling factor for K)


### [StateSpace]

q2 (Process noise)

r2_original (Observation noise in synthetic SS)

r2_pos (Observation position noise in Navigation_Linear State-Space)

r2_vel (Observation velocity noise in Navigation_Linear State-Space)

r2_pr (Observation Psuedo-Range noise in Navigation State-Space)

r2_prr (Observation Psuedo-Range Rate noise in Navigation State-Space)

train_seq_len (Train data sequence length)

train_seq_num (Number of training data sequences)

valid_seq_len (Validation data sequence length)

valid_seq_num (Number of validation data sequences)

test_seq_len (Test data sequence length)

test_seq_num (Number of test data sequences)

model (The State-Space model, options: Synthetic | Navigation | Navigation_Linear)

model_knowledge (The model (filter) knowledge, options: Full | Partial)

trajectory_state (The dataset's state, options: Full | Partial)

rotate_linear_measurements (rotation angle in degrees, for Synthetic SS)

dt (Time-steps difference)

x_dim (State's dimension, for Navigations SS)

n_sat (Number of sattelites to be created, for Navigations SS)

close_sat (Create the sattelites close? (or far) for Navigations SS, options: True | False)

diag_r (Create the mobservation noise covariance matrix R diagonal? for Navigations SS, options: True | False)


### [EKF]

p0 (Initial State's Covariance)

q2 (EKF algorithm's tuneable process noise)


### [Train]

train_iter (Number of training iterations)

valid_period (Validation execution period)

batch_size (Batch size)

beta (Hyper parameter for the Empirical Averaging Loss)

beta_first_iter (Iteration to start averaging in the Empirical Averaging Loss)


### [Train.Kalman]

learning_rate (KalmanNet training learning rate)

weight_decay (KalmanNet training weight decay)


### [Train.Split]

learning_rate (Split-KalmanNet training learning rate)

weight_decay (Split-KalmanNet training weight decay)

alter_period (Split-KalmanNet alternative training period)