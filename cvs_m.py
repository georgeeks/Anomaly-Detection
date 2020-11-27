import scipy.io
import os
import numpy as np
import pandas as pd

current_dir = os.getcwd()
data_path = os.path.join(os.getcwd(),'data')
mat_files = os.listdir(data_path)


for m_file in mat_files:
    mat_file = os.path.join(data_path, m_file)
    mat = scipy.io.loadmat(mat_file)
    mat = {k:v for k, v in mat.items() if k[0] != '_'}
    data = pd.DataFrame({k: pd.Series(v[0]) for k, v in mat.items()})
    save_path = os.path.join(current_dir, 'csv_data')
    data.to_csv(save_path, f"{m_file}.csv")








import scipy.io
st.sidebar.info('**Load examples files:**')

data_path = os.path.join(os.getcwd(),'data')
mat_files = os.listdir(data_path)


for m_file in mat_files:
    mat_file = os.path.join(data_path, select_mat_file)
    data = scipy.io.loadmat(mat_file)
    for i in data:
        if '__' not in i and 'readme' not in i:
            np.savetxt((f"{m_file}.csv"),data[i],delimiter=',')

select_mat_file = st.sidebar.selectbox('mat file', mat_files)
if select_mat_file:
    mat_file = os.path.join(data_path, select_mat_file)
mat = scipy.io.loadmat(mat_file)
st.write(mat)

st.sidebar.info('Shebuti Rayana (2016). ODDS Library [http://odds.cs.stonybrook.edu]. Stony Brook, NY: Stony Brook University, Department of Computer Science.')

#st.write(mat)
#for mat_file in mat_files:
#    mat = scipy.io.loadmat(mat_file)
#    mat_files.append(mat)
#st.write(mat_files)
