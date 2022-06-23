import os
os.makedirs(os.path.join('../../data'), exist_ok = True)
data_file = os.path.join('../../data', 'house_tiny.csv')
with open(data_file) as f:
    f.write('300,why\n')
    f.write('400,qwer\n')