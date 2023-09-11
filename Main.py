from GSSFiltering.model import StateSpaceModel
from GSSFiltering.filtering import Extended_Kalman_Filter
from GSSFiltering.filtering import KalmanNet_Filter, Split_KalmanNet_Filter, KalmanNet_Filter_v2
from GSSFiltering.trainer import Trainer
from GSSFiltering.tester import Tester
import numpy as np
import configparser
import os
import torch
import shutil
import subprocess
import time


# import sys
# sys.path.append('C:/rkn_share-master/')
# from experiments.gnss.gnss import PerformRKN

if not os.path.exists('./.results'):
    os.mkdir('./.results')

config = configparser.ConfigParser()
config.read('./config.ini')


TRAIN=True

if (config['StateSpace']['model'] == "Synthetic"):
    r2 = float(config['StateSpace']['r2_synthetic'])
else:
    r2 = float(config['StateSpace']['r2_pos'])

Trajectory_State = config['StateSpace']['Trajectory_State'].strip("'")

if TRAIN:
    StateSpaceModel(mode='train', knowledge=Trajectory_State).generate_data()
    StateSpaceModel(mode='valid', knowledge=Trajectory_State).generate_data()
StateSpaceModel(mode='test', knowledge=Trajectory_State).generate_data()

train_iter = int(config['Train']['train_iter'])

# S_KalmanNet
test_list = ['100']

loss_list_Kalman = []
loss_list_Kalman_v2 = []
loss_list_Split = []
loss_ekf = []

valid_loss_Kalman = []
valid_loss_Kalman_v2 = []
valid_loss_Split = []

knowledge = config['StateSpace']['model_knowledge'].strip("'")
if TRAIN:
    # KalmanNet
    trainer_kalman = Trainer(
        dnn=KalmanNet_Filter(
            StateSpaceModel(mode='train', knowledge = knowledge)),
        data_path='./.data/StateSpace/train/',
        save_path='(StateSpace) KalmanNet.pt',
        mode=0)
    # trainer_kalman.batch_size = batch_size
    # trainer_kalman.alter_num = alter_num

    # KalmanNet (architecture 2)
    trainer_kalman_v2 = Trainer(
        dnn=KalmanNet_Filter_v2(
            StateSpaceModel(mode='train', knowledge = knowledge)),
        data_path='./.data/StateSpace/train/',
        save_path='(StateSpace, v2) KalmanNet.pt',
        mode=0)
    # trainer_kalman_v2.batch_size = batch_size
    # trainer_kalman_v2.alter_num = alter_num

    # S_KalmanNet
    trainer_split = Trainer(
        dnn=Split_KalmanNet_Filter(
            StateSpaceModel(mode='train', knowledge = knowledge)),
        data_path='./.data/StateSpace/train/',
        save_path='(StateSpace) Split_KalmanNet.pt',
        mode=1)
    # trainer_split.batch_size = batch_size
    # trainer_split.alter_num = alter_num


    for i in range(train_iter):

        trainer_split.train_batch()
        trainer_split.dnn.reset(clean_history=True)
        if trainer_split.train_count % trainer_split.save_num == 0:
            trainer_split.validate(
                Tester(
                        filter = Split_KalmanNet_Filter(
                            StateSpaceModel(mode='valid', knowledge = knowledge)),
                        data_path = './.data/StateSpace/valid/',
                        model_path = './.model_saved/(StateSpace) Split_KalmanNet_' + str(trainer_split.train_count) + '.pt',
                        is_validation=True
                        )
            )
            valid_loss_Split += [trainer_split.valid_loss]

        trainer_kalman.train_batch()
        trainer_kalman.dnn.reset(clean_history=True)
        if trainer_kalman.train_count % trainer_kalman.save_num == 0:
            trainer_kalman.validate(
                Tester(
                        filter = KalmanNet_Filter(
                            StateSpaceModel(mode='valid', knowledge = knowledge)),
                        data_path = './.data/StateSpace/valid/',
                        model_path = './.model_saved/(StateSpace) KalmanNet_' + str(trainer_kalman.train_count) + '.pt',
                        is_validation=True
                        )
            )
            valid_loss_Kalman += [trainer_kalman.valid_loss]

        trainer_kalman_v2.train_batch()
        trainer_kalman_v2.dnn.reset(clean_history=True)
        if trainer_kalman_v2.train_count % trainer_kalman_v2.save_num == 0:
            trainer_kalman_v2.validate(
                Tester(
                        filter = KalmanNet_Filter_v2(
                            StateSpaceModel(mode='valid', knowledge = knowledge)),
                        data_path = './.data/StateSpace/valid/',
                        model_path = './.model_saved/(StateSpace, v2) KalmanNet_' + str(trainer_kalman_v2.train_count) + '.pt',
                        is_validation=True
                        )
            )
            valid_loss_Kalman_v2 += [trainer_kalman_v2.valid_loss]


    validator_ekf = Tester(
                filter = Extended_Kalman_Filter(
                    StateSpaceModel(mode='valid', knowledge = knowledge)),
                data_path = './.data/StateSpace/valid/',
                model_path = 'EKF'
                )
    loss_ekf = [validator_ekf.loss.item()]

    np.save('./.results/valid_loss_ekf.npy', np.array(loss_ekf))
    np.save('./.results/valid_loss_kalman.npy', np.array(valid_loss_Kalman))
    np.save('./.results/valid_loss_kalman_v2.npy', np.array(valid_loss_Kalman_v2))
    np.save('./.results/valid_loss_split.npy', np.array(valid_loss_Split))


tester_ekf = Tester(
            filter = Extended_Kalman_Filter(
                StateSpaceModel(mode='test', knowledge = knowledge)),
            data_path = './.data/StateSpace/test/',
            model_path = 'EKF'
            )
loss_ekf = [tester_ekf.loss.item()]
print(loss_ekf)

for elem in test_list:

    tester_kf = Tester(
                filter = KalmanNet_Filter(
                    StateSpaceModel(mode='test', knowledge = knowledge)),
                data_path = './.data/StateSpace/test/',
                model_path = './.model_saved/(StateSpace) KalmanNet_' + elem + '.pt'
                )
    loss_list_Kalman += [tester_kf.loss.item()]

    tester_kf2 = Tester(
                filter = KalmanNet_Filter_v2(
                    StateSpaceModel(mode='test', knowledge = knowledge)),
                data_path = './.data/StateSpace/test/',
                model_path = './.model_saved/(StateSpace, v2) KalmanNet_' + elem + '.pt'
                )
    loss_list_Kalman_v2 += [tester_kf2.loss.item()]
#
    tester_skf = Tester(
                filter = Split_KalmanNet_Filter(
                    StateSpaceModel(mode='test', knowledge = knowledge)),
                data_path = './.data/StateSpace/test/',
                model_path = './.model_saved/(StateSpace) Split_KalmanNet_' + elem + '.pt'
                )
    loss_list_Split += [tester_skf.loss.item()]

print(loss_ekf)
print(loss_list_Kalman)
print(loss_list_Kalman_v2)
print(loss_list_Split)

# MC_Plot()
# print("Execution of Plots.py completed.")
#
# destination_folder = 'C:/Split_KalmanNet/.results/plots/' + mdl + '/ModelKnowledge_' + knldge + '/LR_' + lr
# if not os.path.exists('C:/Split_KalmanNet/.results/plots/'): os.mkdir('C:/Split_KalmanNet/.results/plots/')
# if not os.path.exists('C:/Split_KalmanNet/.results/plots/' + mdl): os.mkdir( 'C:/Split_KalmanNet/.results/plots/' + mdl)
# if not os.path.exists('C:/Split_KalmanNet/.results/plots/' + mdl + '/ModelKnowledge_' + knldge): os.mkdir('C:/Split_KalmanNet/.results/plots/' + mdl + '/ModelKnowledge_' + knldge)
# if not os.path.exists('C:/Split_KalmanNet/.results/plots/' + mdl + '/ModelKnowledge_' + knldge +  '/LR_' + lr): os.mkdir('C:/Split_KalmanNet/.results/plots/' + mdl + '/ModelKnowledge_' + knldge +  '/LR_' + lr)
#
# # Copy results to new folder:
# source_folder = 'C:\Split_KalmanNet\.results\plots\MonteCarlo'
# new_folder_name = 'SNR='+str(SNR)  # Specify the new folder name
# # Copy the source folder to the destination folder with a new name
# shutil.copytree(source_folder, f"{destination_folder}/{new_folder_name}")
# print(f"Folder copied and renamed to {new_folder_name}")
# time.sleep(5) # Wait for file to be copied and created (!!!)
#
# shutil.copy2('config.ini', destination_folder+'/config.txt')
# time.sleep(5)  # Wait for file to be copied and created (!!!)

# # After All SNRs list is finished:
# PerformRKN(lr, SNRs=SNRs, MainPath=destination_folder)