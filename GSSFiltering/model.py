import torch
import os
import numpy as np
import math, random
from math import pi
from scipy.linalg import sqrtm

import configparser
config = configparser.ConfigParser()

if not os.path.exists('./.data'):
    os.mkdir('./.data')


class GSSModel():
    def __init__(self):
        self.cov_q = None
        self.cov_r = None
        self.x_dim = None
        self.y_dim = None

    def generate_data(self):
        raise NotImplementedError()

    def f(self, current_state):
        raise NotImplementedError()
    def g(self, current_state):
        raise NotImplementedError()
    def Jacobian_f(self, x):
        raise NotImplementedError()
    def Jacobian_g(self, x):
        raise NotImplementedError()
    def next_state(self, current_state):
        config.read('./config.ini')

        if (self.config['model'] == "Synthetic"):
            noise = np.random.multivariate_normal(np.zeros(self.x_dim), self.cov_q)
        elif (self.config['model'] == "Navigation" or self.config['model'] == "Navigation_Linear"):
            dt = self.dt
            dt1, dt2, dt3, dt4, dt5 = dt, dt**2, dt**3, dt**4, dt**5
            if(self.knowledge == 'Full'):
                Q = np.array([[ (1/20)*dt5 , 0 , (1/8)*dt4 , 0 , (1/6)*dt3 , 0 ],
                    [ 0 , (1/20)*dt5 , 0 , (1/8)*dt4 , 0 , (1/6)*dt3 ],
                    [ (1/8)*dt4 , 0 , (1/3)*dt3 , 0 , (1/2)*dt2 , 0 ],
                    [ 0 , (1/8)*dt4 , 0 , (1/3)*dt3 , 0 , (1/2)*dt2 ],
                    [ (1/6)*dt3 , 0 , (1/2)*dt2 , 0 , dt1 , 0 ],
                    [ 0 , (1/6)*dt3 , 0 , (1/2)*dt2 , 0 , dt1]])
            elif(self.knowledge == 'Partial'):
                Q = np.array([[ (1/3)*dt3 , 0 , (1/2)*dt2 , 0],
                    [ 0 , (1/3)*dt3 , 0 , (1/2)*dt2],
                    [ (1/2)*dt2 , 0 , dt1 , 0],
                    [ 0 , (1/2)*dt2 , 0 , dt1]])
            else:
                raise NotImplementedError()

            Q = Q * np.array(self.q2)
            noise = sqrtm(Q) @ np.random.normal(0, 1, size=(self.x_dim,))
        else:
            raise NotImplementedError()

        return self.f(current_state) + torch.tensor(noise, dtype=torch.float).reshape((-1,1))


    def observe(self,current_state):
        noise = np.random.multivariate_normal(np.zeros(self.y_dim), self.cov_r)
        return self.g(current_state) + torch.tensor(noise, dtype=torch.float).reshape((-1,1))

    def Create_Satellites(self, Nsatellites, RandSat = False, Close_sat=False):
        config.read('./config.ini')
        self.sats = []
        for i in range(Nsatellites):
            if RandSat:
                sat = Satellite(random.randint(-500, 500),random.randint(-500, 500),random.randint(50, 100))
            else:
                if Close_sat:
                    # Random option from :
                    # XsatOptions = torch.randint(-300, 300, [1, 30])
                    # YsatOptions = torch.randint(-300, 300, [1, 30])
                    # ZsatOptions = torch.randint(10, 100, [1, 30])

                    XsatOptions = torch.tensor([ 76,   12,  173,   19,  276,  229,  -47,  -10, -295,  273,  175,  -78,
                                                202,  292, -237, -106, -107,  175, -215,  271,  253,  298,  -23, -143,
                                                2,  -47, -241, -158,   28, -226])
                    YsatOptions = torch.tensor([254,  200,  200,   53,  188,  112, -268, -211, -244,  -86,  204,  166,
                                                   25,  167,  -22,  -13,  291,  -67,   50, -268,   32, -224,   98,  -14,
                                                 -187, -189, -289,    5,  -69, -210])
                    ZsatOptions = torch.tensor([67, 55, 30, 56, 24, 64, 86, 58, 40, 78, 11, 51, 35, 99, 75, 68, 34, 54,
                                                60, 41, 69, 53, 34, 21, 13, 77, 47, 16, 92, 87])

                    sat = Satellite(XsatOptions[i],YsatOptions[i],ZsatOptions[i])
                else:
                    sat = Satellite(-2500 + i*1000, -3500 + i*1000, 36000 + i*100)

            self.sats.append(sat)

        self.Nsatellites = Nsatellites
        return self.sats

    def Construct_R(self):
        config.read('./config.ini')
        if (self.config['diag_r'].strip('') == "True"):
            if(self.config['model'] == "Synthetic"):
                return float(self.config['r2_synthetic']) * torch.eye(self.y_dim)
            elif (self.config['model'] == "Navigation"):
                r2_PR = float(self.config['r2_PR'])
                r2_PRR = float(self.config['r2_PRR'])
                if self.knowledge=='Full':
                    return torch.diag(torch.squeeze( torch.cat((r2_PR * torch.ones((1,int(self.y_dim/2))),r2_PRR * torch.ones((1,int(self.y_dim/2)))),dim=1) ))
                elif self.knowledge=='Partial':
                    return torch.diag(torch.squeeze( torch.cat((r2_PR * torch.ones((1,int(self.y_dim/2))),r2_PRR * torch.ones((1,int(self.y_dim/2)))),dim=1) ))
            elif (self.config['model'] == "Navigation_Linear"):
                r2_pos = float(self.config['r2_pos'])
                r2_vel = float(self.config['r2_vel'])
                if self.knowledge=='Full':
                    return torch.diag(torch.squeeze( torch.cat((r2_pos * torch.ones((1,2)),r2_vel * torch.ones((1,2))),dim=1) ))
                elif self.knowledge=='Partial':
                    return torch.diag(torch.squeeze( torch.cat((r2_pos * torch.ones((1, 2)), r2_vel * torch.ones((1, 2))), dim=1)))
        else:
            if(self.config['model'] == "Synthetic"):
                raise NotImplementedError()
            elif (self.config['model'] == "Navigation"):
                r2_PR = float(self.config['r2_PR'])
                r2_PRR = float(self.config['r2_PRR'])
                if self.knowledge=='Full':
                    return torch.block_diag(r2_PR * torch.ones((int(self.y_dim/2),int(self.y_dim/2))), r2_PRR * torch.ones((int(self.y_dim/2),int(self.y_dim/2))))
                elif self.knowledge=='Partial':
                    return torch.block_diag(r2_PR * torch.ones((int(self.y_dim/2),int(self.y_dim/2))), r2_PRR * torch.ones((int(self.y_dim/2),int(self.y_dim/2))))
            elif (self.config['model'] == "Navigation_Linear"):
                r2_pos = float(self.config['r2_pos'])
                r2_vel = float(self.config['r2_vel'])
                if self.knowledge=='Full':
                    return torch.block_diag(r2_pos * torch.ones((2,2)), r2_vel * torch.ones((2,2)))
                elif self.knowledge=='Partial':
                    return torch.block_diag(r2_pos * torch.ones((2,2)), r2_vel * torch.ones((2,2)))

class Satellite:
    # Initialize the object with the x, y, and z coordinates of the satellite
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    # Function to create the Pseudo-Range (PR) and Pseudo-Range Rate (PRR)
    def createPR_PRR(self, Ref):
        # Calculate the PR using the coordinates of the satellite and the reference station
        PR = np.sqrt((Ref[1,:]-self.x)**2 + (Ref[2,:]-self.y)**2 + (0-self.z)**2)
        # Calculate the PRR using the coordinates of the satellite and the reference station
        PRR = (Ref[3,:]*(Ref[1,:]-self.x) + Ref[4,:]*(Ref[2,:]-self.y) + 0*(0-self.z))/PR
        # Assign the calculated PR and PRR to the object
        self.PR = np.squeeze(PR)
        self.PRR = np.squeeze(PRR)


class StateSpaceModel(GSSModel):
    def __init__(self, mode='train', knowledge='Full'):
        config.read('./config.ini')
        super().__init__()
        self.config = config['StateSpace']
        # mode = 'train' or 'valid' or 'test'
        self.mode = mode
        self.knowledge = knowledge
        self.sats = None

        if self.mode not in ['train', 'valid', 'test']:
            raise ValueError('Possible mode = ["train", "valid", "test"]')
        if(self.config['model'] == "Synthetic"):
            if self.knowledge =='Full':
                self.alpha = 0.9
                self.beta = 1.1
                self.phi = 0.1*pi
                self.delta = 0.01
                self.a = 1
                self.b = 1
                self.c = 0
            elif self.knowledge =='Partial':
                self.alpha = 1
                self.beta = 1
                self.phi = 0
                self.delta = 0
                self.a = 1
                self.b = 1
                self.c = 0
            else:
                raise NotImplementedError()
            self.x_dim = 2
            self.y_dim = 2
        elif (self.config['model'] == "Navigation"):
            self.y_dim = int(self.config['n_Sat']) * 2
            self.dt = float(self.config['dt'])
            r2_PR = float(self.config['r2_PR'])
            r2_PRR = float(self.config['r2_PRR'])
            if knowledge=='Full':
                self.x_dim = int(self.config['x_dim'])
            elif knowledge=='Partial':
                self.x_dim = int(self.config['x_dim']) - 2 # no Acc_x Acc_y
        elif (self.config['model'] == "Navigation_Linear"):
            r2_pos = float(self.config['r2_pos'])
            r2_vel = float(self.config['r2_vel'])
            if knowledge=='Full':
                self.x_dim = int(self.config['x_dim'])
            elif knowledge=='Partial':
                self.x_dim = int(self.config['x_dim']) - 2 # no Acc_x Acc_y
            self.y_dim = 4 # Position, Velocity, X and Y
            self.dt = float(self.config['dt'])

        self.q2 = float(self.config['q2'])
        self.cov_q = self.q2 * torch.eye(self.x_dim)
        self.cov_r = GSSModel.Construct_R(self)
        # self.v = float(self.config['v'])
        # self.r2 = torch.mul(self.q2, self.v)

        if(self.config['model'] == "Synthetic"):
            # self.init_state = torch.tensor([1., 0.]).reshape((-1, 1))
            # self.init_cov = torch.zeros((self.x_dim, self.x_dim))
            self.init_state = torch.tensor([0., 0.]).reshape((-1, 1))
            self.init_cov = float(config['EKF']['P0']) * torch.eye(self.x_dim)
        elif (self.config['model'] == "Navigation" or self.config['model'] == "Navigation_Linear"):
            dt = self.dt
            if knowledge=='Full':
                self.F = torch.tensor([[1, 0, dt, 0, 0.5 * dt ** 2, 0],
                                       [0, 1, 0, dt, 0, 0.5 * dt ** 2],
                                       [0, 0, 1, 0, dt, 0],
                                       [0, 0, 0, 1, 0, dt],
                                       [0, 0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0, 1]])
            elif knowledge=='Partial':
                self.F = torch.tensor([[1, 0, dt, 0],
                                       [0, 1, 0, dt],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])
            self.sats = GSSModel.Create_Satellites(self, Nsatellites = int(self.config['n_Sat']), RandSat=False, Close_sat=((self.config['Close_Sat'].strip(''))=="True")) # Max 30 sattelites
            # self.init_state = torch.tensor([0.01, 0.01, 1, 1, 1, 1]).reshape(-1,1)
            # self.init_state = (torch.rand(self.x_dim)*2-1).reshape(-1, 1)
            # self.init_cov = torch.zeros((self.x_dim, self.x_dim))
            self.init_state = (torch.zeros(self.x_dim)).reshape(-1, 1)
            self.init_cov = float(config['EKF']['P0']) * torch.eye(self.x_dim)
        else:
            raise NotImplementedError()

    def generate_data(self):
        config.read('./config.ini')
        self.save_path = './.data/StateSpace/'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)      

        if self.mode == 'train':
            if (self.config['model'] == "Synthetic"):
                self.seq_len = int(self.config['train_seq_len'])
            elif (self.config['model'] == "Navigation" or self.config['model'] == "Navigation_Linear"):
                self.seq_len = int(int(self.config['train_seq_len']) / self.dt)
            else:
                raise NotImplementedError()
            self.num_data = int(self.config['train_seq_num'])
            self.save_path += 'train/'
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)      
        elif self.mode == 'valid':
            if (self.config['model'] == "Synthetic"):
                self.seq_len = int(self.config['valid_seq_len'])
            elif (self.config['model'] == "Navigation" or self.config['model'] == "Navigation_Linear"):
                self.seq_len = int(int(self.config['valid_seq_len']) / self.dt)
            else:
                raise NotImplementedError()
            self.num_data = int(self.config['valid_seq_num'])
            self.save_path += 'valid/'
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path) 
        elif self.mode == 'test':
            if (self.config['model'] == "Synthetic"):
                self.seq_len = int(self.config['test_seq_len'])
            elif (self.config['model'] == "Navigation" or self.config['model'] == "Navigation_Linear"):
                self.seq_len = int(int(self.config['test_seq_len']) / self.dt)
            else:
                raise NotImplementedError()
            self.num_data = int(self.config['test_seq_num'])
            self.save_path += 'test/'
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path) 
        else:
            raise NotImplementedError()

        state_mtx = torch.zeros((self.num_data, self.x_dim, self.seq_len))
        obs_mtx = torch.zeros((self.num_data, self.y_dim, self.seq_len))

        with torch.no_grad():
            for i in range(self.num_data):

                if i % 100 == 0:
                    print(f'Saving {i} / {self.num_data} at {self.save_path}')
                state_tmp = torch.zeros((self.x_dim, self.seq_len))
                obs_tmp = torch.zeros((self.y_dim, self.seq_len))
                state_last = torch.clone(self.init_state)

                for j in range(self.seq_len):
                    x = self.next_state(state_last)
                    state_last = torch.clone(x)
                    y = self.observe(x)
                    state_tmp[:,j] = x.reshape(-1)
                    obs_tmp[:,j] = y.reshape(-1)
                
                state_mtx[i] = state_tmp
                obs_mtx[i] = obs_tmp
        
        torch.save(state_mtx, self.save_path + 'state.pt')
        torch.save(obs_mtx, self.save_path + 'obs.pt')

    def f(self, x):
        config.read('./config.ini')
        if (self.config['model'] == "Synthetic"):
            return self.alpha*torch.sin(self.beta*x+self.phi)+self.delta
        else:
            return torch.matmul(self.F, x)
    
    def g(self, x):
        config.read('./config.ini')

        if (self.config['model'] == "Synthetic"):
            return self.a*(self.b*x+self.c)**2
        elif (self.config['model'] == "Navigation"):
            x = x.reshape(-1)
            meas = []
            # Psuedo-Range
            for sat in self.sats:
                # Calculate the PR using the coordinates of the satellite and the reference station
                PR = np.sqrt((x[0] - sat.x) ** 2 + (x[1] - sat.y) ** 2 + (0 - sat.z) ** 2)
                # Stack the measurements
                meas.append(PR)
            # Psuedo-Range Rate
            for sat in self.sats:
                # Calculate the PRR using the coordinates of the satellite and the reference station
                PRR = (x[2] * (x[0] - sat.x) + x[3] * (x[1] - sat.y) + 0 * (0 - sat.z)) / PR
                # Stack the measurements
                meas.append(PRR)
            return torch.tensor([meas]).reshape((-1, 1))
        elif (self.config['model'] == "Navigation_Linear"):
            x = x.reshape(-1)
            # Checks if its in Generate Data or not
            if('num_data' in dir(self)):
                theta = float(self.config['Rotate_Linear_Measurements']) * pi / 180
                rotate = torch.tensor(
                    [[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])
            else:
                rotate = torch.eye(2)
            pos_xy = rotate @ torch.tensor([[x[0]],[x[1]]]) # Rotated Position
            vel_xy = rotate @ torch.tensor([[x[2]], [x[3]]])  # Rotated Position
            meas = []
            meas.append(pos_xy[0]) # Position X
            meas.append(pos_xy[1]) # Position Y
            meas.append(vel_xy[0]) # Velocity X
            meas.append(vel_xy[1]) # Velocity Y
            return torch.tensor([meas]).reshape((-1, 1))
        else:
            raise NotImplementedError()

    def Jacobian_f(self, x):
        config.read('./config.ini')
        if (self.config['model'] == "Synthetic"):
            return torch.diag(torch.squeeze( self.alpha * self.beta * torch.cos(self.beta * x + self.phi) ))
        else:
            return self.F
        
    def Jacobian_g(self, x):
        config.read('./config.ini')

        if (self.config['model'] == "Synthetic"):
            return torch.diag(torch.squeeze(2*self.a*self.b*(self.b*x+self.c)))
        elif (self.config['model'] == "Navigation"):
            # Prepare Structures:
            nSat = len(self.sats)
            H = torch.zeros([nSat * 2, x.shape[0]])
            Pos_x = x[0]
            Pos_y = x[1]
            Pos_z = torch.zeros(Pos_x.shape)
            # Block Jacobian
            for ind, sat in enumerate(self.sats):
                Xsat = sat.x
                Ysat = sat.y
                Zsat = sat.z
                Sat_Distance = torch.sqrt((Xsat - Pos_x) ** 2 + (Ysat - Pos_y) ** 2 + (Zsat - Pos_z) ** 2)
                x_hat = (Pos_x - Xsat) / Sat_Distance
                y_hat = (Pos_y - Ysat) / Sat_Distance
                zero_padd2 = torch.zeros(1,2)  # For dh/dx Equation
                if self.knowledge == 'Partial':
                    H[ind + 0, :] = torch.hstack((x_hat.reshape(-1,1), y_hat.reshape(-1,1), zero_padd2))
                    H[ind + nSat, :] = torch.hstack((zero_padd2, x_hat.reshape(-1,1), y_hat.reshape(-1,1)))
                elif self.knowledge == 'Full':
                    H[ind + 0, :] = torch.hstack((x_hat.reshape(-1,1), y_hat.reshape(-1,1), zero_padd2, zero_padd2))
                    H[ind + nSat, :] = torch.hstack((zero_padd2, x_hat.reshape(-1,1), y_hat.reshape(-1,1), zero_padd2))
            return H
        elif(self.config['model'] == "Navigation_Linear"):
            if self.knowledge == 'Partial':
                H = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
            elif self.knowledge == 'Full':
                H = torch.tensor([[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0.]])
            return H
        else:
            raise NotImplementedError()
