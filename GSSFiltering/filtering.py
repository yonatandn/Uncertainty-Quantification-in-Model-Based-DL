from GSSFiltering.dnn import DNN_KalmanNet_GSS, DNN_SKalmanNet_GSS, KNet_architecture_v2
from GSSFiltering.model import GSSModel
import torch
import torch.nn.functional as F

import configparser
config = configparser.ConfigParser()

class Extended_Kalman_Filter():
    
    def __init__(self, GSSModel:GSSModel):
        config.read('./config.ini')
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.init_state = GSSModel.init_state
        # self.Q = GSSModel.cov_q
        # self.R = GSSModel.cov_r
        self.Q = float(config['EKF']['q2']) * torch.eye(self.x_dim)
        self.R = GSSModel.Construct_R()
        self.R = torch.diag(torch.diag(self.R)) # The Filter Does not know about the off diagonal

        # self.init_cov = torch.zeros((self.x_dim, self.x_dim))
        self.init_cov = float(config['EKF']['P0']) * torch.eye(self.x_dim)
        self.state_history = self.init_state.detach().clone()
        self.reset(clean_history=True)   

    def reset(self, clean_history=False):
        number_of_states_tracked=4
        self.state_post = self.init_state.detach().clone()
        self.cov_post = self.init_cov.detach().clone()
        # self.state_history = torch.cat((self.state_history, self.state_post), axis=1)
        if clean_history:
            # Clean history for a new Trajectory, with initial states
            self.state_history = self.init_state.detach().clone()
            self.cov_history = torch.unsqueeze(self.init_cov.detach().clone(), dim=2)
            self.cov_trace_history = torch.zeros((1,))
            self.cov_pred_history = torch.unsqueeze(self.init_cov.detach().clone(), dim=2)
            self.cov_pred_byK_opt1_history = torch.unsqueeze(self.init_cov[:number_of_states_tracked,:number_of_states_tracked].detach().clone(), dim=2)
            self.cov_pred_byK_opt2_history = torch.unsqueeze(self.init_cov[:number_of_states_tracked,:number_of_states_tracked].detach().clone(), dim=2)
            self.cov_post_optA_history = torch.unsqueeze(self.init_cov.detach().clone(), dim=2)
            self.cov_post_optB_history = torch.unsqueeze(self.init_cov.detach().clone(), dim=2)

        self.state_history = torch.cat((self.state_history, self.state_post), axis=1)
        self.cov_history = torch.cat((self.cov_history, torch.unsqueeze(self.init_cov.detach().clone(), dim=2)), axis=2)
        self.cov_pred_history = torch.cat((self.cov_pred_history, torch.unsqueeze(self.init_cov.detach().clone(), dim=2)), axis=2)
        self.cov_pred_byK_opt1_history = torch.cat((self.cov_pred_byK_opt1_history, torch.unsqueeze(self.init_cov[:number_of_states_tracked,:number_of_states_tracked].detach().clone(), dim=2)), axis=2)
        self.cov_pred_byK_opt2_history = torch.cat((self.cov_pred_byK_opt2_history, torch.unsqueeze(self.init_cov[:number_of_states_tracked,:number_of_states_tracked].detach().clone(), dim=2)), axis=2)
        self.cov_post_optA_history = torch.cat((self.cov_post_optA_history, torch.unsqueeze(self.init_cov.detach().clone(), dim=2)), axis=2)
        self.cov_post_optB_history = torch.cat((self.cov_post_optB_history, torch.unsqueeze(self.init_cov.detach().clone(), dim=2)), axis=2)

    def filtering(self, observation):
        config.read('./config.ini')
        with torch.no_grad():
            # print(self.GSSModel.r2)
            # observation: column vector
            x_last = self.state_post
            x_predict = self.GSSModel.f(x_last)

            y_predict = self.GSSModel.g(x_predict)
            residual = observation - y_predict

            F_jacob = self.GSSModel.Jacobian_f(x_last)
            H_jacob = self.GSSModel.Jacobian_g(x_predict)
            cov_pred = (F_jacob @ self.cov_post @ torch.transpose(F_jacob, 0, 1)) + self.Q

            K_gain = cov_pred @ torch.transpose(H_jacob, 0, 1) @ \
                torch.linalg.inv(H_jacob@cov_pred@torch.transpose(H_jacob, 0, 1) + self.R)

            x_post = x_predict + (K_gain @ residual)

            cov_post = (torch.eye(self.x_dim) - K_gain @ H_jacob) @ cov_pred
            cov_trace = torch.trace(cov_post)

            self.pk = cov_pred

            self.state_post = x_post.detach().clone()
            self.cov_post = cov_post.detach().clone()
            self.state_history = torch.cat((self.state_history, x_post.clone()), axis=1)    
            self.cov_trace_history = torch.cat((self.cov_trace_history, cov_trace.reshape(-1).clone()))
            self.cov_history = torch.cat((self.cov_history, torch.unsqueeze(cov_post, dim=2).clone()), axis=2)

            try:
                # For Uncertainty Analysis:
                if (config['StateSpace']['model'] == "Synthetic"):
                    scz = 2  # State's Chopped Size
                else:
                    scz = 4 # State's Chopped Size
                H = H_jacob[:, :scz]
                K = K_gain[:scz, :]
                Ht = torch.transpose(H, 0, 1)
                Htilde = torch.linalg.inv(Ht @ H)
                HtildeHt = Htilde @ Ht
                invI_HK = torch.linalg.inv(torch.eye(self.y_dim) - H @ K)
                HKR = H @ K @ self.R
                HHtilde = H @ Htilde
                invKH_I = torch.linalg.inv(K @ H - torch.eye(scz))

                cov_pred_byK_opt1 = HtildeHt @ invI_HK @ HKR @ HHtilde  # Option 1
                cov_pred_byK_opt2 = - invKH_I @ K @ self.R @ H @ Htilde # Option 2
                cov_post_optA = (torch.eye(self.x_dim) - K_gain @ H_jacob) @ cov_pred # Option A
                cov_post_optB = (torch.eye(self.x_dim) - K_gain @ H_jacob) @ cov_pred @ (torch.eye(self.x_dim) - K_gain @ H_jacob).T + K_gain @ self.R @ K_gain.T # Option B

                self.cov_pred_history = torch.cat((self.cov_pred_history, torch.unsqueeze(cov_pred, dim=2).clone()), axis=2)
                self.cov_pred_byK_opt1_history = torch.cat((self.cov_pred_byK_opt1_history, torch.unsqueeze(cov_pred_byK_opt1, dim=2).clone()), axis=2)
                self.cov_pred_byK_opt2_history = torch.cat((self.cov_pred_byK_opt2_history, torch.unsqueeze(cov_pred_byK_opt2, dim=2).clone()), axis=2)
                self.cov_post_optA_history = torch.cat((self.cov_post_optA_history, torch.unsqueeze(cov_post_optA, dim=2).clone()), axis=2)
                self.cov_post_optB_history = torch.cat((self.cov_post_optB_history, torch.unsqueeze(cov_post_optB, dim=2).clone()), axis=2)

                # ImHHtildeHT_summed = sum(torch.diag(torch.eye(self.y_dim) - H @ Htilde @ H.T))
                # if ImHHtildeHT_summed>10: print("EKF: I - (H*Htidle*H.T) = {}".format(ImHHtildeHT_summed))
            except:
                print("Failed at Uncertainty Analysis of EKF")

class KalmanNet_Filter():
    def __init__(self, GSSModel:GSSModel):
        
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.kf_net = DNN_KalmanNet_GSS(self.x_dim, self.y_dim)
        self.init_state = GSSModel.init_state
        self.reset(clean_history=True)

        # For Uncertainty analysis:
        # self.R = float(config['EKF']['r2']) * torch.eye(self.y_dim)
        self.R = GSSModel.Construct_R()
        self.R = torch.diag(torch.diag(self.R)) # The Filter Does not know about the off diagonal

    def reset(self, clean_history=False):
        scz = 4 # chopped_state_size
        self.dnn_first = True
        self.kf_net.initialize_hidden()
        self.state_post = self.init_state.detach().clone()
        if clean_history:
            self.state_history = self.init_state.detach().clone()  
            self.cov_trace_history = torch.zeros((1,))
            self.cov_pred_byK_opt1_history = torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)
            self.cov_pred_byK_opt2_history = torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)
            self.cov_post_byK_optA_history = torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)
            self.cov_post_byK_optB_history = torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)
        self.state_history = torch.cat((self.state_history, self.state_post), axis=1)
        self.cov_pred_byK_opt1_history = torch.cat((self.cov_pred_byK_opt1_history, torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)), axis=2)
        self.cov_pred_byK_opt2_history = torch.cat((self.cov_pred_byK_opt2_history, torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)), axis=2)
        self.cov_post_byK_optA_history = torch.cat((self.cov_post_byK_optA_history, torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)), axis=2)
        self.cov_post_byK_optB_history = torch.cat((self.cov_post_byK_optB_history, torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)), axis=2)

    def filtering(self, observation):
        # observation: column vector
        config.read('./config.ini')
        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.GSSModel.f(x_last)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.GSSModel.g(x_predict)

        ## input 1: x_{k-1 | k-1} - x_{k-1 | k-2}
        state_inno = self.state_post_past - self.state_pred_past
        ## input 2: residual
        residual = observation - y_predict
        ## input 3: x_k - x_{k-1}
        diff_state = self.state_post - self.state_post_past
        ## input 4: y_k - y_{k-1}
        diff_obs = observation - self.obs_past

        K_gain = self.kf_net(state_inno, residual, diff_state, diff_obs)

        x_post = x_predict + (K_gain @ residual)

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), axis=1)

        # For Uncertainty Analysis:
        try:
            H_jacob = self.GSSModel.Jacobian_g(x_predict)
            if (config['StateSpace']['model'] == "Synthetic"):
                scz = 2  # State's Chopped Size
            else:
                scz = 4  # State's Chopped Size
            H = H_jacob[:,:scz]
            K = K_gain[:scz,:]
            I = torch.eye(scz)
            # Ht = torch.transpose(H, 0, 1)
            Ht = H.T
            Htilde = torch.linalg.inv(Ht @ H)
            HtildeHt = Htilde @ H.T
            invI_HK = torch.linalg.inv(torch.eye(self.y_dim) - H @ K)
            HKR = H @ K @ self.R
            HHtilde = H @ Htilde
            invKH_I = torch.linalg.inv(K @ H - torch.eye(scz))

            cov_pred_byK_opt1 = HtildeHt @ invI_HK @ HKR @ HHtilde
            cov_pred_byK_opt2 = - invKH_I @ K @ self.R @ H @ Htilde

            cov_post_byK_optA = (I - K @ H) @ cov_pred_byK_opt2
            cov_post_byK_optB = (I - K @ H) @ cov_pred_byK_opt2 @ (I - K @ H).T + K @ self.R @ K.T

            self.cov_pred_byK_opt1_history = torch.cat((self.cov_pred_byK_opt1_history, torch.unsqueeze(cov_pred_byK_opt1, dim=2).clone()),axis=2)
            self.cov_pred_byK_opt2_history = torch.cat((self.cov_pred_byK_opt2_history, torch.unsqueeze(cov_pred_byK_opt2, dim=2).clone()),axis=2)

            self.cov_post_byK_optA_history = torch.cat((self.cov_post_byK_optA_history, torch.unsqueeze(cov_post_byK_optA, dim=2).clone()), axis=2)
            self.cov_post_byK_optB_history = torch.cat((self.cov_post_byK_optB_history, torch.unsqueeze(cov_post_byK_optB, dim=2).clone()), axis=2)


            # ImHHtildeHT_summed = sum(torch.diag(torch.eye(self.y_dim) - H @ Htilde @ H.T))
            # if ImHHtildeHT_summed>10: print("KNet V1: I - (H*Htidle*H.T) = {}".format(ImHHtildeHT_summed))
        except:
            print("Failed at Uncertainty Analysis of KalmanNet V1")

class KalmanNet_Filter_v2():
    def __init__(self, GSSModel:GSSModel):
        
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.kf_net = KNet_architecture_v2(self.x_dim, self.y_dim)
        self.init_state = GSSModel.init_state
        self.reset(clean_history=True)

        # For Uncertainty analysis:
        self.R = GSSModel.Construct_R()
        self.R = torch.diag(torch.diag(self.R)) # The Filter Does not "know" about the off-diagonal

    def reset(self, clean_history=False):
        scz = 4  # chopped_state_size
        self.dnn_first = True
        self.kf_net.initialize_hidden()
        self.state_post = self.init_state.detach().clone()
        if clean_history:
            self.state_history = self.init_state.detach().clone()
            self.cov_trace_history = torch.zeros((1,))
            self.cov_pred_byK_opt1_history = torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)
            self.cov_pred_byK_opt2_history = torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)
            self.cov_post_byK_optA_history = torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)
            self.cov_post_byK_optB_history = torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)
        self.state_history = torch.cat((self.state_history, self.state_post), axis=1)
        self.cov_pred_byK_opt1_history = torch.cat((self.cov_pred_byK_opt1_history, torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)), axis=2)
        self.cov_pred_byK_opt2_history = torch.cat((self.cov_pred_byK_opt2_history, torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)), axis=2)
        self.cov_post_byK_optA_history = torch.cat((self.cov_post_byK_optA_history, torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)), axis=2)
        self.cov_post_byK_optB_history = torch.cat((self.cov_post_byK_optB_history, torch.unsqueeze(self.GSSModel.init_cov[:scz, :scz].detach().clone(), dim=2)), axis=2)

    def filtering(self, observation):
        config.read('./config.ini')
        # observation: column vector

        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.GSSModel.f(x_last)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.GSSModel.g(x_predict)

        ## input 1: x_{k-1 | k-1} - x_{k-1 | k-2}
        state_inno = self.state_post_past - self.state_pred_past
        ## input 2: residual
        residual = observation - y_predict
        ## input 3: x_k - x_{k-1}
        diff_state = self.state_post - self.state_post_past
        ## input 4: y_k - y_{k-1}
        diff_obs = observation - self.obs_past

        K_gain = self.kf_net(diff_obs, residual, diff_state, state_inno)

        x_post = x_predict + (K_gain @ residual)

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), axis=1)

        # For Uncertainty Analysis:
        try:
            H_jacob = self.GSSModel.Jacobian_g(x_predict)
            if (config['StateSpace']['model'] == "Synthetic"):
                scz = 2  # State's Chopped Size
            else:
                scz = 4  # State's Chopped Size
            H = H_jacob[:, :scz]
            K = K_gain[:scz, :]
            I = torch.eye(scz)
            # Ht = torch.transpose(H, 0, 1)
            Ht = H.T
            Htilde = torch.linalg.inv(Ht @ H)
            HtildeHt = Htilde @ H.T
            invI_HK = torch.linalg.inv(torch.eye(self.y_dim) - H @ K)
            HKR = H @ K @ self.R
            HHtilde = H @ Htilde
            invKH_I = torch.linalg.inv(K @ H - torch.eye(scz))

            cov_pred_byK_opt1 = HtildeHt @ invI_HK @ HKR @ HHtilde
            cov_pred_byK_opt2 = - invKH_I @ K @ self.R @ H @ Htilde

            cov_post_byK_optA = (I - K @ H) @ cov_pred_byK_opt2
            cov_post_byK_optB = (I - K @ H) @ cov_pred_byK_opt2 @ (I - K @ H).T + K @ self.R @ K.T

            self.cov_pred_byK_opt1_history = torch.cat((self.cov_pred_byK_opt1_history, torch.unsqueeze(cov_pred_byK_opt1, dim=2).clone()), axis=2)
            self.cov_pred_byK_opt2_history = torch.cat((self.cov_pred_byK_opt2_history, torch.unsqueeze(cov_pred_byK_opt2, dim=2).clone()), axis=2)

            self.cov_post_byK_optA_history = torch.cat((self.cov_post_byK_optA_history, torch.unsqueeze(cov_post_byK_optA, dim=2).clone()), axis=2)
            self.cov_post_byK_optB_history = torch.cat((self.cov_post_byK_optB_history, torch.unsqueeze(cov_post_byK_optB, dim=2).clone()), axis=2)

        except:
            print("Failed at Uncertainty Analysis of KalmanNet V2")

        # ImHHtildeHT_summed = sum(torch.diag(torch.eye(self.y_dim) - H @ Htilde @ H.T))
        # if ImHHtildeHT_summed>10: print("KNet v2: I - (H*Htidle*H.T) = {}".format(ImHHtildeHT_summed))

class Split_KalmanNet_Filter():
    def __init__(self, GSSModel:GSSModel):
        
        self.x_dim = GSSModel.x_dim
        self.y_dim = GSSModel.y_dim
        self.GSSModel = GSSModel

        self.kf_net = DNN_SKalmanNet_GSS(self.x_dim, self.y_dim)
        self.init_state = GSSModel.init_state
        self.init_cov = GSSModel.init_cov
        self.reset(clean_history=True)

        # For Uncertainty analysis:
        # self.R = float(config['EKF']['r2']) * torch.eye(self.y_dim)
        self.R = GSSModel.Construct_R()
        self.R = torch.diag(torch.diag(self.R)) # The Filter Does not know about the off diagonal

    def reset(self, clean_history=False):
        self.dnn_first = True
        self.kf_net.initialize_hidden()
        self.state_post = self.init_state.detach().clone()
        self.cov_post = torch.unsqueeze(self.init_cov.detach().clone(), dim=2)
        if clean_history:
            self.state_history = self.init_state.detach().clone()
            self.cov_history = torch.unsqueeze(self.init_cov.detach().clone(), dim=2)
            self.cov_pred_history = torch.unsqueeze(self.init_cov.detach().clone(), dim=2)
            self.cov_pred_byK_opt1_history = torch.unsqueeze(self.init_cov[:4,:4].detach().clone(), dim=2)
            self.cov_pred_byK_opt2_history = torch.unsqueeze(self.init_cov[:4,:4].detach().clone(), dim=2)
            self.cov_post_byK_optA_history = torch.unsqueeze(self.init_cov[:4,:4].detach().clone(), dim=2)
            self.cov_post_byK_optB_history = torch.unsqueeze(self.init_cov[:4,:4].detach().clone(), dim=2)
        self.state_history = torch.cat((self.state_history, self.state_post), axis=1)
        self.cov_history = torch.cat((self.cov_history, self.cov_post), axis=2)
        self.cov_pred_history = torch.cat((self.cov_pred_history, torch.unsqueeze(self.init_cov.detach().clone(), dim=2)), axis=2)
        self.cov_pred_byK_opt1_history = torch.cat((self.cov_pred_byK_opt1_history, torch.unsqueeze(self.init_cov[:4,:4].detach().clone(), dim=2)), axis=2)
        self.cov_pred_byK_opt2_history = torch.cat((self.cov_pred_byK_opt2_history, torch.unsqueeze(self.init_cov[:4,:4].detach().clone(), dim=2)), axis=2)
        self.cov_post_byK_optA_history = torch.cat((self.cov_post_byK_optA_history, torch.unsqueeze(self.init_cov[:4,:4].detach().clone(), dim=2)), axis=2)
        self.cov_post_byK_optB_history = torch.cat((self.cov_post_byK_optB_history, torch.unsqueeze(self.init_cov[:4,:4].detach().clone(), dim=2)), axis=2)


    def filtering(self, observation):
        # observation: column vector
        config.read('./config.ini')
        if self.dnn_first:
            self.state_post_past = self.state_post.detach().clone()

        x_last = self.state_post
        x_predict = self.GSSModel.f(x_last)

        if self.dnn_first:
            self.state_pred_past = x_predict.detach().clone()
            self.obs_past = observation.detach().clone()

        y_predict = self.GSSModel.g(x_predict)

        ## input 1: x_{k-1 | k-1} - x_{k-1 | k-2}
        state_inno = self.state_post_past - self.state_pred_past
        ## input 2: residual
        residual = observation - y_predict
        ## input 3: x_k - x_{k-1}
        diff_state = self.state_post - self.state_post_past
        ## input 4: y_k - y_{k-1}
        diff_obs = observation - self.obs_past
        ## input 6: Jacobian
        H_jacob = self.GSSModel.Jacobian_g(x_predict)     
        ## input 5: linearization error
        # linearization_error = H_jacob@x_predict
        linearization_error = y_predict - H_jacob@x_predict
        H_jacob_in = H_jacob.reshape((-1,1))
        # H_jacob_in = F.normalize(H_jacob_in, p=2, dim=0, eps=1e-12)
        linearization_error_in = linearization_error
        # linearization_error_in = F.normalize(linearization_error, p=2, dim=0, eps=1e-12)
        (Pk, Sk) = self.kf_net(state_inno, residual, diff_state, diff_obs, linearization_error_in, H_jacob_in)

        cov_pred = Pk

        K_gain = Pk @ torch.transpose(H_jacob, 0, 1) @ Sk

        x_post = x_predict + (K_gain @ residual)
        cov_post = (torch.eye(self.x_dim) - K_gain @ H_jacob) @ Pk # (I - K * H) Pk

        self.dnn_first = False
        self.state_pred_past = x_predict.detach().clone()
        self.state_post_past = self.state_post.detach().clone()
        self.obs_past = observation.detach().clone()
        self.state_post = x_post.detach().clone()
        self.state_history = torch.cat((self.state_history, x_post.clone()), axis=1)
        self.cov_history = torch.cat((self.cov_history, torch.unsqueeze(cov_post, dim=2).clone()), axis=2)

        # For Uncertainty Analysis:
        try:
            if (config['StateSpace']['model'] == "Synthetic"):
                scz = 2  # State's Chopped Size
            else:
                scz = 4  # State's Chopped Size
            H = H_jacob[:,:scz]
            K = K_gain[:scz,:]
            I = torch.eye(scz)
            # Ht = torch.transpose(H, 0, 1)
            Ht = H.T
            Htilde = torch.linalg.inv(Ht @ H)
            HtildeHt = Htilde @ H.T
            invI_HK = torch.linalg.inv(torch.eye(self.y_dim) - H @ K)
            HKR = H @ K @ self.R
            HHtilde = H @ Htilde
            invKH_I = torch.linalg.inv(K @ H - torch.eye(scz))

            cov_pred_byK_opt1 = HtildeHt @ invI_HK @ HKR @ HHtilde
            cov_pred_byK_opt2 = - invKH_I @ K @ self.R @ H @ Htilde

            cov_post_byK_optA = (I - K @ H) @ cov_pred_byK_opt2
            cov_post_byK_optB = (I - K @ H) @ cov_pred_byK_opt2 @ (I - K @ H).T + K @ self.R @ K.T

            self.cov_pred_history = torch.cat((self.cov_pred_history, torch.unsqueeze(cov_pred, dim=2).clone()), axis=2)
            self.cov_pred_byK_opt1_history = torch.cat((self.cov_pred_byK_opt1_history, torch.unsqueeze(cov_pred_byK_opt1, dim=2).clone()), axis=2)
            self.cov_pred_byK_opt2_history = torch.cat((self.cov_pred_byK_opt2_history, torch.unsqueeze(cov_pred_byK_opt2, dim=2).clone()), axis=2)

            self.cov_post_byK_optA_history = torch.cat((self.cov_post_byK_optA_history, torch.unsqueeze(cov_post_byK_optA, dim=2).clone()), axis=2)
            self.cov_post_byK_optB_history = torch.cat((self.cov_post_byK_optB_history, torch.unsqueeze(cov_post_byK_optB, dim=2).clone()), axis=2)

            # ImHHtildeHT_summed = sum(torch.diag(torch.eye(self.y_dim) - H @ Htilde @ H.T))
            # if ImHHtildeHT_summed>10: print("Split: I - (H*Htidle*H.T) = {}".format(ImHHtildeHT_summed))
            
        except Exception as exc:
            print("Failed at Uncertainty Analysis of Split KalmanNet")
