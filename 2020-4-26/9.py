import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing

# 读取泰坦尼克号生存数据
train_url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls"
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
df_data = pd.read_excel(train_path)


# 数据预处理函数
def prepare_data(temp_data):
    temp_data = temp_data.drop(['name'], axis=1)  # 删除姓名项
    temp_data['age'] = temp_data['age'].fillna(temp_data['age'].mean())  # 年龄为空的填入均值
    temp_data['fare'] = temp_data['fare'].fillna(temp_data['fare'].mean())  # 票价为空的填入均值
    temp_data['embarked'] = temp_data['embarked'].fillna('S')  # 登船港口为空填入南安普顿
    temp_data['sex'] = temp_data['sex'].map({'female': 0, 'male': 1}).astype(int)  # 把性别转换为数值
    temp_data['embarked'] = temp_data['embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)  # 把登船港口转换为数值

    ndarray = temp_data.values
    features = ndarray[:, 1:]  # 特征
    label = ndarray[:, 0]  # 标签

    # 特征归一化，值为0-1之间
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    features = minmax_scale.fit_transform(features)

    return features, label


# 选择数据
selected_col = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
select_data = df_data[selected_col]
select_data = select_data.sample(frac=1)
x, y = prepare_data(select_data)

# 划分数据集
train_num = int(len(x) * 0.8)
x_train, y_train = x[:train_num], y[:train_num]
x_test, y_test = x[train_num:], y[train_num:]

model = tf.keras.Sequential([
    # input(n, 7) * kernel(7, 64) --> output(n, 64)
    tf.keras.layers.Dense(units=64, input_shape=(7,), activation=tf.nn.relu),
    # input(n, 64) * kernel(64, 32) --> output(n, 32)
    tf.keras.layers.Dense(units=32, activation=tf.nn.sigmoid),
    # input(n, 32) * kernel(32, 1) --> output(n, 1)
    tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid)
])

model.summary()
model.compile(
    optimizer=tf.optimizers.Adam(0.001),
    loss=tf.losses.binary_crossentropy,
    metrics=['accuracy'],
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss'),
    tf.keras.callbacks.TerminateOnNaN()
]

model.fit(
    x=x_train,
    y=y_train,
    batch_size=40,
    epochs=15,
    validation_split=0.2,
    callbacks=callbacks
)

model.evaluate(x_test, y_test)

# jack和rose的数据
jack_info = [0, 'Jack', 3, 'male', 23, 1, 0, 5.0000, 'S']
rose_info = [1, 'Rose', 1, 'female', 20, 1, 0, 100.0000, 'S']

all_passenger = select_data.append(pd.DataFrame([jack_info, rose_info], columns=selected_col))
x_features, _ = prepare_data(all_passenger)
survey = model.predict(x_features)
all_passenger.insert(len(all_passenger.columns), 'survey', survey)
print(all_passenger[-5:])
