import os
import mne
import glob
import pandas as pd
from feature_extraction import transform_egg_into_fixed_length


# read edf file
def read_edf_file(file_path):
    eeg_data, tot_channels_name = read_edf_and_eeg_channels_transformation(file_path)

    print(eeg_data.shape)

    cha_ann_df = pd.read_csv('sample.csv')
    print(cha_ann_df.head())

    edfbasepath = os.path.basename(file_path)

    # check if the filename us in csv file to get its label class
    subj_defected_chann = cha_ann_df[cha_ann_df['data'].str.contains(edfbasepath)]

    if subj_defected_chann.shape[0] > 1:
        print("Duplicate found")
        # defected_channels_list = subj_defected_chann['channel_defected'].iloc[0]
        defected_channels_list = str(subj_defected_chann.iloc[0][1])
        print(subj_defected_chann.iloc[0][0], defected_channels_list)

    elif subj_defected_chann.shape[0] > 0:
        print("Single record found")
        print(subj_defected_chann)

        defected_channels_list = str(subj_defected_chann.iloc[0][1])
        print(subj_defected_chann.iloc[0][0], defected_channels_list)
    else:
        print("No defected channels found")
        defected_channels_list = ""

    return eeg_data, defected_channels_list, tot_channels_name


# read the edf transform the channels into other montage space
def read_edf_and_eeg_channels_transformation(edf_file_path):
    edf_file = mne.io.read_raw_edf(edf_file_path, eog=['FP1', 'FP2', 'F3', 'F4',
                                                       'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                                                       'T3', 'T4', 'T5', 'T6', 'PZ', 'FZ', 'CZ', 'A1', 'A2'
                                                       ],
                                   exclude=['Add_lead1', 'Add_lead2', 'Add_lead3', 'Add_lead4', 'Add_lead5',
                                            'Add_lead6',
                                            'Add_lead7', 'Add_lead8'],
                                   verbose='error', preload=True)
    edf_file_down_sampled = edf_file.resample(250, npad="auto")  # set sampling frequency to 250 Hz
    ed = edf_file_down_sampled.to_data_frame(picks=None, index=None,copy=True, start=None, stop=None)  # converting into dataframe

    print(ed.head())
    Fp1_Fp7 = (ed.loc[:, 'FP1']) - (ed.loc[:, 'F7'])
    FP2_F8 = (ed.loc[:, 'FP2']) - (ed.loc[:, 'F8'])
    F7_T3 = (ed.loc[:, 'F7']) - (ed.loc[:, 'T3'])
    F8_T4 = (ed.loc[:, 'F8']) - (ed.loc[:, 'T4'])
    T3_T5 = (ed.loc[:, 'T3']) - (ed.loc[:, 'T5'])
    T4_T6 = (ed.loc[:, 'T4']) - (ed.loc[:, 'T6'])
    T5_O1 = (ed.loc[:, 'T5']) - (ed.loc[:, 'O1'])
    T6_O2 = (ed.loc[:, 'T6']) - (ed.loc[:, 'O2'])
    A1_T3 = (ed.loc[:, 'A1']) - (ed.loc[:, 'T3'])
    T4_A2 = (ed.loc[:, 'T4']) - (ed.loc[:, 'A2'])
    T3_C3 = (ed.loc[:, 'T3']) - (ed.loc[:, 'C3'])
    C4_T4 = (ed.loc[:, 'C4']) - (ed.loc[:, 'T4'])
    C3_CZ = (ed.loc[:, 'C3']) - (ed.loc[:, 'CZ'])
    CZ_C4 = (ed.loc[:, 'CZ']) - (ed.loc[:, 'C4'])
    FP1_F3 = (ed.loc[:, 'FP1']) - (ed.loc[:, 'F3'])
    FP2_F4 = (ed.loc[:, 'FP2']) - (ed.loc[:, 'F4'])
    F3_C3 = (ed.loc[:, 'F3']) - (ed.loc[:, 'C3'])
    F4_C4 = (ed.loc[:, 'F4']) - (ed.loc[:, 'C4'])
    C3_P3 = (ed.loc[:, 'C3']) - (ed.loc[:, 'P3'])
    C4_P4 = (ed.loc[:, 'C4']) - (ed.loc[:, 'P4'])
    P3_O1 = (ed.loc[:, 'P3']) - (ed.loc[:, 'O1'])
    P4_O2 = (ed.loc[:, 'P4']) - (ed.loc[:, 'O2'])
    data = {
        'Fp1-Fp7': Fp1_Fp7,
        'FP2-F8': FP2_F8,
        'F7-T3': F7_T3,
        'F8-T4': F8_T4,
        'T3-T5': T3_T5,
        'T4-T6': T4_T6,
        'T5-O1': T5_O1,
        'T6-O2': T6_O2,
        'A1-T3': A1_T3,
        'T4-A2': T4_A2,
        'T3-C3': T3_C3,
        'C4-T4': C4_T4,
        'C3-CZ': C3_CZ,
        'CZ-C4': CZ_C4,
        'FP1-F3': FP1_F3,
        'FP2-F4': FP2_F4,
        'F3-C3': F3_C3,
        'F4-C4': F4_C4,
        'C3-P3': C3_P3,
        'C4-P4': C4_P4,
        'P3-O1': P3_O1,
        'P4-O2': P4_O2

    }

    total_chann_name = ['Fp1-Fp7', 'FP2-F8', 'F7-T3', 'F8-T4', 'T3-T5', 'T4-T6', 'T5-O1', 'T6-O2',
                        'A1-T3', 'T4-A2', 'T3-C3', 'C4-T4', 'C3-CZ',
                        'CZ-C4', 'FP1-F3', 'FP2-F4', 'F3-C3', 'F4-C4', 'C3-P3', 'C4-P4', 'P3-O1',
                        'P4-O2'
                        ]
    new_data_frame = pd.DataFrame(data,
                                  columns=total_chann_name)

    return new_data_frame.values, total_chann_name


# get training data
def get_training_data():
    train_data = []
    for filepath in glob.iglob('data/*.edf'):

        subject_eeg, df_channels, tot_channels_name = read_edf_file(filepath)

        fixed_length_vector, df_channels, tot_channels_name, drop_status = transform_egg_into_fixed_length(subject_eeg,
                                                                                                           df_channels,
                                                                                                           tot_channels_name)

        if not drop_status:
            for channel_data in fixed_length_vector:
                train_data.append(channel_data)

    print(train_data)

    return train_data


