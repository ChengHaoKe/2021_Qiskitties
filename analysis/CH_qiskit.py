import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import QSVM
from qiskit.aqua.utils import split_dataset_to_data_and_labels, map_label_to_class_name
from qiskit.ml.datasets import ad_hoc_data, sample_ad_hoc_data
from qiskit.circuit.library import ZZFeatureMap
from qiskit.aqua.components.multiclass_extensions import AllPairs
from qiskit.aqua.utils.dataset_helper import get_feature_dimension
from qiskit.ml.datasets import wine
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.feature_maps import RawFeatureVector
from qiskit.circuit.library import TwoLocal
import qiskit
import os
print(qiskit.__version__)


cwd = os.getcwd()
print(cwd)
if not '2021_Qiskitties' in cwd:
    os.chdir('2021_Qiskitties')
elif not cwd.endswith('2021_Qiskitties'):
    os.chdir('..')
print(os.getcwd())


seed = 10599
aqua_globals.random_seed = seed


df = pd.read_csv('int_data/CH_data.csv')

print(df['threat_level'].value_counts())
print(df.info())

df_cleaned = df.dropna()

# Drop location variables
df_cleaned = df_cleaned.drop(['id', 'name', 'date', 'citylist.1', 'Geographic_Area.1', 'City.1',
                              'citylist', 'lencity', 'state'], axis=1)

df_cleaned['Median_Income'] = pd.to_numeric(df_cleaned['Median Income'], errors='coerce')
df_cleaned['poverty_rate'] = pd.to_numeric(df_cleaned['poverty_rate'], errors='coerce')
df_cleaned['percent_completed_hs'] = pd.to_numeric(df_cleaned['percent_completed_hs'], errors='coerce')
racecol = ['share_white', 'share_black', 'share_native_american', 'share_asian', 'share_hispanic']
for r in racecol:
    df_cleaned[r] = df_cleaned[r].str.replace(r"[a-zA-Z]|\(|\)", '', regex=True)
    df_cleaned[r] = df_cleaned[r].apply(lambda x: float(x) if x != '' else np.nan)


# recode threat level
df_cleaned.drop(['Median Income'], axis=1, inplace=True)
df_cleaned['threat'] = df_cleaned['threat_level'].replace({'undetermined': 'other'})

# Response
# response_name = 'threat_level'
response_name = 'threat'
print(df_cleaned['threat_level'].value_counts())
print(df_cleaned['threat'].value_counts())


# One hot encoding
enc = OneHotEncoder(handle_unknown='ignore')

df_enc = df_cleaned.copy()

# Drop response column since it is not a physical attribute of the car
df_enc = df_enc.drop([response_name], axis=1)

# Get column names of categorical and numerical variables
cat_names = df_enc.select_dtypes(include='object').columns
num_names = df_enc.select_dtypes(include=np.number).columns

# Encode categorical variables
enc_columns = pd.get_dummies(df_cleaned[cat_names], drop_first=True)

# Concatenate encoded columns to numerical columns
df_enc = pd.concat([df_enc[num_names], enc_columns], axis=1)

df_enc.head()


X = df_enc.copy()
y = df_cleaned[response_name]


feature_dim = 2
sample_total, training_input, test_input, class_labels = ad_hoc_data(
    training_size=20,
    test_size=10,
    n=feature_dim,
    gap=0.3,
    plot_data=True
)
extra_test_data = sample_ad_hoc_data(sample_total, 10, n=feature_dim)
datapoints, class_to_label = split_dataset_to_data_and_labels(extra_test_data)
print(class_to_label)


feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=2, entanglement='linear')
qsvm = QSVM(feature_map, training_input, test_input, datapoints[0])

backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=2, seed_simulator=seed, seed_transpiler=seed)

result = qsvm.run(quantum_instance)

print(f'Testing success ratio: {result["testing_accuracy"]}')
print()
print('Prediction from datapoints set:')
print(f'  ground truth: {map_label_to_class_name(datapoints[1], qsvm.label_to_class)}')
print(f'  prediction:   {result["predicted_classes"]}')
predicted_labels = result["predicted_labels"]
print(f'  success rate: {100 * np.count_nonzero(predicted_labels == datapoints[1]) / len(predicted_labels)}%')

n = 2  # dimension of each data point
sample_Total, training_input, test_input, class_labels = wine(training_size=24,
                                                              test_size=6, n=n, plot_data=True)
temp = [test_input[k] for k in test_input]
total_array = np.concatenate(temp)

aqua_globals.random_seed = 10598

backend = BasicAer.get_backend('qasm_simulator')
feature_map = ZZFeatureMap(feature_dimension=get_feature_dimension(training_input),
                           reps=2, entanglement='linear')
svm = QSVM(feature_map, training_input, test_input, total_array,
           multiclass_extension=AllPairs())
quantum_instance = QuantumInstance(backend, shots=2,
                                   seed_simulator=aqua_globals.random_seed,
                                   seed_transpiler=aqua_globals.random_seed)

result = svm.run(quantum_instance)
for k, v in result.items():
    print(f'{k} : {v}')


feature_dim = np.shape(X)[1]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

training_input = X_train.copy()
test_input = X_test.copy()

total_array = np.concatenate([test_input[k] for k in test_input])

class_labels = y.unique()
training_size = 100
test_size = 25


sample_train, sample_test, label_train, label_test = \
    train_test_split(X, y, test_size=0.2, random_state=7)

# Now we standardize for gaussian around 0 with unit variance
std_scale = StandardScaler().fit(sample_train)
sample_train = std_scale.transform(sample_train)
sample_test = std_scale.transform(sample_test)

# Now reduce number of features to number of qubits
pca = PCA(n_components=n).fit(sample_train)
sample_train = pca.transform(sample_train)
sample_test = pca.transform(sample_test)

# Scale to the range (-1,+1)
samples = np.append(sample_train, sample_test, axis=0)
minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
sample_train = minmax_scale.transform(sample_train)
sample_test = minmax_scale.transform(sample_test)


training_input = {key: (sample_train[label_train == key, :])[:training_size]
                  for k, key in enumerate(class_labels)}
test_input = {key: (sample_test[label_test == key, :])[:test_size]
              for k, key in enumerate(class_labels)}

aqua_globals.random_seed = 10598

backend = BasicAer.get_backend('qasm_simulator')
feature_map = ZZFeatureMap(feature_dimension=get_feature_dimension(training_input),
                           reps=2, entanglement='linear')
svm = QSVM(feature_map, training_input, test_input, total_array,
           multiclass_extension=AllPairs())
quantum_instance = QuantumInstance(backend, shots=1024,
                                   seed_simulator=aqua_globals.random_seed,
                                   seed_transpiler=aqua_globals.random_seed)

result = svm.run(quantum_instance)
for k, v in result.items():
    print(f'{k} : {v}')


seed = 1376
aqua_globals.random_seed = seed

# Use Wine data set for training and test data
feature_dim = 4  # dimension of each data point
_, training_input, test_input, _ = wine(training_size=12,
                                        test_size=4,
                                        n=feature_dim)

feature_map = RawFeatureVector(feature_dimension=feature_dim)
vqc = VQC(COBYLA(maxiter=100),
          feature_map,
          TwoLocal(feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3),
          training_input,
          test_input)
result = vqc.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                 shots=2, seed_simulator=seed, seed_transpiler=seed))

print('Testing accuracy: {:0.2f}'.format(result['testing_accuracy']))
