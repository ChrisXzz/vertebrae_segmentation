import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression

csv_file = 'vertebrae_shape_statistics.csv'

data = pd.read_csv(csv_file)

data_train = data[data['dataset']=='train']

cervical_data_train = data_train[data_train['group label']==1]
thoracic_data_train = data_train[data_train['group label']==2]
lumbar_data_train = data_train[data_train['group label']==3]


## gap from pre - cervical

cervical_data_gap = cervical_data_train.dropna(subset=['gap', 'pre gap'])

cervical_gap = np.array(cervical_data_gap['gap']).reshape(-1, 1)
cervical_gap_pre = np.array(cervical_data_gap['pre gap']).reshape(-1, 1)

reg = LinearRegression().fit(cervical_gap_pre, cervical_gap)
# print(reg.score(cervical_gap_pre, cervical_gap))
# print(reg.coef_, reg.intercept_)
print('cervical:\n gap = pre gap * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## gap from pre - thoracic

thoracic_data_gap = thoracic_data_train.dropna(subset=['gap', 'pre gap'])

thoracic_gap = np.array(thoracic_data_gap['gap']).reshape(-1, 1)
thoracic_gap_pre = np.array(thoracic_data_gap['pre gap']).reshape(-1, 1)

reg = LinearRegression().fit(thoracic_gap_pre, thoracic_gap)
# print(reg.score(thoracic_gap_pre, thoracic_gap))
# print(reg.coef_, reg.intercept_)
print('\nthoracic:\n gap = pre gap * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## gap from pre - lumbar

lumbar_data_gap = lumbar_data_train.dropna(subset=['gap', 'pre gap'])

lumbar_gap = np.array(lumbar_data_gap['gap']).reshape(-1, 1)
lumbar_gap_pre = np.array(lumbar_data_gap['pre gap']).reshape(-1, 1)

reg = LinearRegression().fit(lumbar_gap_pre, lumbar_gap)
# print(reg.score(lumbar_gap_pre, lumbar_gap))
# print(reg.coef_, reg.intercept_)
print('\nlumbar:\n gap = pre gap * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## gap from pre = general

data_gap = data_train.dropna(subset=['gap', 'pre gap'])

gap = np.array(data_gap['gap']).reshape(-1, 1)
gap_pre = np.array(data_gap['pre gap']).reshape(-1, 1)


reg = LinearRegression().fit(gap_pre, gap)
# print(reg.score(gap_pre, gap))
# print(reg.coef_, reg.intercept_)
print('\ngeneral:\n gap = pre gap * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## gap from next - cervical

cervical_data_gap = cervical_data_train.dropna(subset=['gap', 'next gap'])

cervical_gap = np.array(cervical_data_gap['gap']).reshape(-1, 1)
cervical_gap_next = np.array(cervical_data_gap['next gap']).reshape(-1, 1)

reg = LinearRegression().fit(cervical_gap_next, cervical_gap)
# print(reg.score(cervical_gap_next, cervical_gap))
# print(reg.coef_, reg.intercept_)
print('\ncervical:\n gap = next gap * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## gap from next - thoracic

thoracic_data_gap = thoracic_data_train.dropna(subset=['gap', 'next gap'])

thoracic_gap = np.array(thoracic_data_gap['gap']).reshape(-1, 1)
thoracic_gap_next = np.array(thoracic_data_gap['next gap']).reshape(-1, 1)

reg = LinearRegression().fit(thoracic_gap_next, thoracic_gap)
# print(reg.score(thoracic_gap_next, thoracic_gap))
# print(reg.coef_, reg.intercept_)
print('\nthoracic:\n gap = next gap * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## gap from next - lumbar

lumbar_data_gap = lumbar_data_train.dropna(subset=['gap', 'next gap'])

lumbar_gap = np.array(lumbar_data_gap['gap']).reshape(-1, 1)
lumbar_gap_next = np.array(lumbar_data_gap['next gap']).reshape(-1, 1)

reg = LinearRegression().fit(lumbar_gap_next, lumbar_gap)
# print(reg.score(lumbar_gap_next, lumbar_gap))
# print(reg.coef_, reg.intercept_)
print('\nlumbar:\n gap = next gap * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## gap from next - general

data_gap = data_train.dropna(subset=['gap', 'next gap'])

gap = np.array(data_gap['gap']).reshape(-1, 1)
gap_next = np.array(data_gap['next gap']).reshape(-1, 1)

reg = LinearRegression().fit(gap_next, gap)
# print(reg.score(gap_next, gap))
# print(reg.coef_, reg.intercept_)
print('\ngeneral:\n gap = next gap * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## gap from two sides - cervical

cervical_data_gap = cervical_data_train.dropna(subset=['pre gap', 'gap', 'next gap'])

cervical_gap = np.array(cervical_data_gap['gap']).reshape(-1, 1)
cervical_gap_pre = np.array(cervical_data_gap['pre gap']).reshape(-1, 1)
cervical_gap_next = np.array(cervical_data_gap['next gap']).reshape(-1, 1)
cervical_gap_neighbor = np.concatenate((cervical_gap_pre, cervical_gap_next), axis=1)

reg = LinearRegression().fit(cervical_gap_neighbor, cervical_gap)
# print(reg.score(cervical_gap_neighbor, cervical_gap))
# print(reg.coef_, reg.intercept_)
print('\ncervical:\n gap = pre gap * {:.2f} + next gap * {:.2f} + {:.2f}'.format(
                           reg.coef_[0][0], reg.coef_[0][1], reg.intercept_[0]))

## gap from two sides - thoracic

thoracic_data_gap = thoracic_data_train.dropna(subset=['pre gap', 'gap', 'next gap'])

thoracic_gap = np.array(thoracic_data_gap['gap']).reshape(-1, 1)
thoracic_gap_pre = np.array(thoracic_data_gap['pre gap']).reshape(-1, 1)
thoracic_gap_next = np.array(thoracic_data_gap['next gap']).reshape(-1, 1)
thoracic_gap_neighbor = np.concatenate((thoracic_gap_pre, thoracic_gap_next), axis=1)

reg = LinearRegression().fit(thoracic_gap_neighbor, thoracic_gap)
# print(reg.score(thoracic_gap_neighbor, thoracic_gap))
# print(reg.coef_, reg.intercept_)
print('\nthoracic:\n gap = pre gap * {:.2f} + next gap * {:.2f} + {:.2f}'.format(
                           reg.coef_[0][0], reg.coef_[0][1], reg.intercept_[0]))

## gap from two sides - lumbar

lumbar_data_gap = lumbar_data_train.dropna(subset=['pre gap',  'gap', 'next gap'])

lumbar_gap = np.array(lumbar_data_gap['gap']).reshape(-1, 1)
lumbar_gap_pre = np.array(lumbar_data_gap['pre gap']).reshape(-1, 1)
lumbar_gap_next = np.array(lumbar_data_gap['next gap']).reshape(-1, 1)
lumbar_gap_neighbor = np.concatenate((lumbar_gap_pre, lumbar_gap_next), axis=1)

reg = LinearRegression().fit(lumbar_gap_neighbor, lumbar_gap)
# print(reg.score(lumbar_gap_neighbor, lumbar_gap))
# print(reg.coef_, reg.intercept_)
print('\nlumbar:\n gap = pre gap * {:.2f} + next gap * {:.2f} + {:.2f}'.format(
                           reg.coef_[0][0], reg.coef_[0][1], reg.intercept_[0]))

## gap from two sides - general

data_gap = data_train.dropna(subset=['pre gap', 'gap', 'next gap'])

gap = np.array(data_gap['gap']).reshape(-1, 1)
gap_pre = np.array(data_gap['pre gap']).reshape(-1, 1)
gap_next = np.array(data_gap['next gap']).reshape(-1, 1)
gap_neighbor = np.concatenate((gap_pre, gap_next), axis=1)

reg = LinearRegression().fit(gap_neighbor, gap)
# print(reg.score(gap_neighbor, gap))
# print(reg.coef_, reg.intercept_)
print('\ngeneral:\n gap = pre gap * {:.2f} + next gap * {:.2f} + {:.2f}'.format(
                           reg.coef_[0][0], reg.coef_[0][1], reg.intercept_[0]))

## size from pre - cervical

cervical_data_size = cervical_data_train.dropna(subset=['size of pre', 'size'])

cervical_size = np.array(cervical_data_size['size']).reshape(-1, 1)
cervical_size_pre = np.array(cervical_data_size['size of pre']).reshape(-1, 1)

reg = LinearRegression().fit(cervical_size_pre, cervical_size)
# print(reg.score(cervical_size_pre, cervical_size))
# print(reg.coef_, reg.intercept_)
print('\ncervical:\n size = size of pre * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## size from pre - thoracic

thoracic_data_size = thoracic_data_train.dropna(subset=['size of pre', 'size'])

thoracic_size = np.array(thoracic_data_size['size']).reshape(-1, 1)
thoracic_size_pre = np.array(thoracic_data_size['size of pre']).reshape(-1, 1)

reg = LinearRegression().fit(thoracic_size_pre, thoracic_size)
# print(reg.score(thoracic_size_pre, thoracic_size))
# print(reg.coef_, reg.intercept_)
print('\nthoracic:\n size = size of pre * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## size from pre - lumbar

lumbar_data_size = lumbar_data_train.dropna(subset=['size of pre', 'size'])

lumbar_size = np.array(lumbar_data_size['size']).reshape(-1, 1)
lumbar_size_pre = np.array(lumbar_data_size['size of pre']).reshape(-1, 1)

reg = LinearRegression().fit(lumbar_size_pre, lumbar_size)
# print(reg.score(lumbar_size_pre, lumbar_size))
# print(reg.coef_, reg.intercept_)
print('\nlumbar:\n size = size of pre * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## size from pre - general
data_size = data_train.dropna(subset=['size', 'size of pre'])

size = np.array(data_size['size']).reshape(-1, 1)
size_pre = np.array(data_size['size of pre']).reshape(-1, 1)

reg = LinearRegression().fit(size_pre, size)
# print(reg.score(size_pre, size))
# print(reg.coef_, reg.intercept_)
print('\ngeneral:\n size = size of pre * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## size from next - cervical

cervical_data_size = cervical_data_train.dropna(subset=['size of next', 'size'])

cervical_size = np.array(cervical_data_size['size']).reshape(-1, 1)
cervical_size_next = np.array(cervical_data_size['size of next']).reshape(-1, 1)

reg = LinearRegression().fit(cervical_size_next, cervical_size)
# print(reg.score(cervical_size_next, cervical_size))
# print(reg.coef_, reg.intercept_)
print('\ncervical:\n size = size of next * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## size from next - thoracic

thoracic_data_size = thoracic_data_train.dropna(subset=['size of next', 'size'])

thoracic_size = np.array(thoracic_data_size['size']).reshape(-1, 1)
thoracic_size_next = np.array(thoracic_data_size['size of next']).reshape(-1, 1)

reg = LinearRegression().fit(thoracic_size_next, thoracic_size)
# print(reg.score(thoracic_size_next, thoracic_size))
# print(reg.coef_, reg.intercept_)
print('\nthoracic:\n size = size of next * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## size from next - lumbar

lumbar_data_size = lumbar_data_train.dropna(subset=['size of next', 'size'])

lumbar_size = np.array(lumbar_data_size['size']).reshape(-1, 1)
lumbar_size_next = np.array(lumbar_data_size['size of next']).reshape(-1, 1)

reg = LinearRegression().fit(lumbar_size_next, lumbar_size)
# print(reg.score(lumbar_size_next, lumbar_size))
# print(reg.coef_, reg.intercept_)
print('\nlumbar:\n size = size of next * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

## size from next - general
data_size = data_train.dropna(subset=['size', 'size of next'])

size = np.array(data_size['size']).reshape(-1, 1)
size_next = np.array(data_size['size of next']).reshape(-1, 1)

reg = LinearRegression().fit(size_next, size)
# print(reg.score(size_next, size))
# print(reg.coef_, reg.intercept_)
print('\ngeneral:\n size = size of next * {:.2f} + {:.2f}'.format(reg.coef_[0][0], reg.intercept_[0]))

