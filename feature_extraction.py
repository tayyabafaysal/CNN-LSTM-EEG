import numpy as np

# transformed eeg into fixed length
def transform_egg_into_fixed_length(raw_data, defected_channels_list, tot_channels_name):
    # transpose the array
    raw_data = raw_data.T
    print("After transposing: ", raw_data.shape)

    # how long should take eeg
    time_duration = 10000

    subject_eeg = raw_data

    # if the timediff is larger then threshold then drop
    drop_status = False

    if raw_data.shape[1] > time_duration:
        raw_data = raw_data[:, :time_duration]
        print("Fixed Length vector", raw_data.shape)
    else:
        # print("Do some concatenation")

        time_diff = int(time_duration - raw_data.shape[1])
        if time_diff > 10000:
            drop_status = True
        else:
            drop_status = False
        

    # now the vector is fixed, its turn to transformed the fixed vector
    fixed_length_vector = raw_data
    df_channels = defected_channels_list

    # print(len(fixed_length_vector), len(fixed_length_vector[0]))
    # add a zero  value column to the end of list
    fixed_length_vector = np.c_[fixed_length_vector, np.zeros(subject_eeg.shape[0])]
    # print(len(fixed_length_vector), len(fixed_length_vector[0]))

    # only first index
    if len(df_channels) > 0:
        # df_channels = df_channels[0]
        df_channels = df_channels.split(',')

    # if check for at least 1 channel
    if len(df_channels) > 0:

        # label defected channel wise eeg
        for dfc in df_channels:

      
            try:
                # search dfc in whole channels list
                ch_idx = tot_channels_name.index(dfc)

                fixed_length_vector[ch_idx][-1] = 1
            except Exception as e:
                print("Channel not found. Please find the attached traceback .", e)

    return fixed_length_vector, defected_channels_list, tot_channels_name, drop_status
