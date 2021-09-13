from pdb import set_trace
import mne
import pandas as pd
import numpy as np
import math
import os
import h5py
import csv


def readDatafromPath(path):

	size=[]
	for file in os.listdir(path):
		if '.edf' in file:
			f=os.path.join(path, file)
			print(f)
			edf_file = mne.io.read_raw_edf(f, montage = None, eog = ['FP1', 'FP2', 'F3', 'F4',
			'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
			'T3', 'T4', 'T5', 'T6', 'PZ', 'FZ', 'CZ', 'A1', 'A2'
			], verbose = 'error', preload = True)
			edf_file_down_sampled = edf_file.resample(250, npad = "auto")# set sampling frequency to 250 Hz
			ed = edf_file_down_sampled.to_data_frame(picks = None, index = None, scaling_time = 1000.0, scalings = None,
			copy = True, start = None, stop = None)# converting into dataframe
			Fp1_Fp7 = (ed.loc[: , 'FP1']) - (ed.loc[: , 'F7'])
			FP2_F8 = (ed.loc[: , 'FP2']) - (ed.loc[: , 'F8'])
			F7_T3 = (ed.loc[: , 'F7']) - (ed.loc[: , 'T3'])
			F8_T4 = (ed.loc[: , 'F8']) - (ed.loc[: , 'T4'])
			T3_T5 = (ed.loc[: , 'T3']) - (ed.loc[: , 'T5'])
			T4_T6 = (ed.loc[: , 'T4']) - (ed.loc[: , 'T6'])
			T5_O1 = (ed.loc[: , 'T5']) - (ed.loc[: , 'O1'])
			T6_O2 = (ed.loc[: , 'T6']) - (ed.loc[: , 'O2'])
			A1_T3 = (ed.loc[: , 'A1']) - (ed.loc[: , 'T3'])
			T4_A2 = (ed.loc[: , 'T4']) - (ed.loc[: , 'A2'])
			T3_C3 = (ed.loc[: , 'T3']) - (ed.loc[: , 'C3'])
			C4_T4 = (ed.loc[: , 'C4']) - (ed.loc[: , 'T4'])
			C3_CZ = (ed.loc[: , 'C3']) - (ed.loc[: , 'CZ'])
			CZ_C4 = (ed.loc[: , 'CZ']) - (ed.loc[: , 'C4'])
			FP1_F3 = (ed.loc[: , 'FP1']) - (ed.loc[: , 'F3'])
			FP2_F4 = (ed.loc[: , 'FP2']) - (ed.loc[: , 'F4'])
			F3_C3 = (ed.loc[: , 'F3']) - (ed.loc[: , 'C3'])
			F4_C4 = (ed.loc[: , 'F4']) - (ed.loc[: , 'C4'])
			C3_P3 = (ed.loc[: , 'C3']) - (ed.loc[: , 'P3'])
			C4_P4 = (ed.loc[: , 'C4']) - (ed.loc[: , 'P4'])
			P3_O1 = (ed.loc[: , 'P3']) - (ed.loc[: , 'O1'])
			P4_O2 = (ed.loc[: , 'P4']) - (ed.loc[: , 'O2'])
			data = {
			'Fp1_Fp7': Fp1_Fp7,
			'FP2_F8': FP2_F8,
			'F7_T3': F7_T3,
			'F8_T4': F8_T4,
			'T3_T5': T3_T5,
			'T4_T6': T4_T6,
			'T5_O1': T5_O1,
			'T6_O2': T6_O2,
			'A1_T3': A1_T3,
			'T4_A2': T4_A2,
			'T3_C3': T3_C3,
			'C4_T4': C4_T4,
			'C3_CZ': C3_CZ,
			'CZ_C4': CZ_C4,
			'FP1_F3': FP1_F3,
			'FP2_F4': FP2_F4,
			'F3_C3': F3_C3,
			'F4_C4': F4_C4,
			'C3_P3': C3_P3,
			'C4_P4': C4_P4,
			'P3_O1': P3_O1,
			'P4_O2': P4_O2
			}
			new_data_frame = pd.DataFrame(data, columns = ['Fp1_Fp7', 'FP2_F8', 'F7_T3', 'F8_T4', 'T3_T5', 'T4_T6', 'T5_O1', 'T6_O2', 'A1_T3', 'T4_A2', 'T3_C3', 'C4_T4', 'C3_CZ',
			'CZ_C4', 'FP1_F3', 'FP2_F4', 'F3_C3', 'F4_C4', 'C3_P3', 'C4_P4', 'P3_O1', 'P4_O2'
			])
			fs = edf_file_down_sampled.info['sfreq']
			[row, col] = new_data_frame.shape
			size.append(row//(fs*60))
	return size

print('starting')
normal = readDatafromPath(path = "normal/train")
abnormal = readDatafromPath(path = "abnormal/train")

normal_eval = readDatafromPath(path = "normal/eval")
abnormal_eval = readDatafromPath(path = "abnormal/eval")
print('training labels have been loaded')


with open('hist.csv', 'a', newline='') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(normal)
    wr.writerow(abnormal)
    wr.writerow(normal_eval)
    wr.writerow(abnormal_eval)