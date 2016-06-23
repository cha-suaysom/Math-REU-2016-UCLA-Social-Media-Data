import pandas # Works with data in table format
import numpy as np
import math
import pickle

raw_data = pickle.load(open('pandas_data_barc.pkl','rb'))
location_accurate_raw_data = raw_data[raw_data['gps_precision']==10.0]


size = len(location_accurate_raw_data["gps_precision"])


sampled_raw_data = location_accurate_raw_data.sample(frac = 0.5)
print(size, len(sampled_raw_data))

pickle.dump(sampled_raw_data, open('05_pandas_data_barc10.pkl','wb'))

pickle.dump(sampled_raw_data['text'].tolist(), open('05raw_text_data_barc10.pkl','wb'))


