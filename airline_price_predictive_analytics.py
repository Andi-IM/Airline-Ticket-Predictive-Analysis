# -*- coding: utf-8 -*-
"""Airline Price Predictive Analytics.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ikhtTIUbRE3wUE5wYb-obruLhaOwaUOA

# Airline Price Predictive Analytics

## About

Proyek ini akan menganalisa dataset penerbangan dari platform [Kaggle](kaggle.com) dan membuat model prediksi harga tiket pesawat.

# Import Libraries
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns
import tensorflow as tf
import seaborn as sns

# Run this cell and select the kaggle.json file downloaded
# from the Kaggle account settings page.
from google.colab import files
files.upload()

# Let's make sure the kaggle.json file is present.
!ls -lha kaggle.json

# Commented out IPython magic to ensure Python compatibility.
# The Kaggle API client expects this file to be in ~/.kaggle,
# so move it there.
# %mkdir -p ~/.kaggle
# %cp kaggle.json ~/.kaggle/

# This permissions change avoids a warning on Kaggle tool startup.
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d shubhambathwal/flight-price-prediction

!unzip flight-price-prediction.zip -d .

"""# Data Loading"""

airlines = pd.read_csv('Clean_Dataset.csv')
airlines

"""Output kode di atas memberikan informasi sebagai berikut:
- Terdapat 300.153 baris (records atau jumlah pengamatan) dalam dataset.
- Terdapat 12 kolom yaitu: Unnamed: 0, airline, flight, source_city, departure_time, stops, arrival_time, destination_city, class, duration, days_left, price.

# Explanatory Data Analysis

## Informasi dataset
"""

airlines.info()

"""Dari output terlihat bahwa:
- Terdapat 8 kolom dengan tipe object, yaitu: airline, flight, source_city, departure_time, stops, arrival_time, destination_city, class. Kolom ini merupakan categorical features (fitur non-numerik).
- Terdapat 1 kolom dengan tipe data float64 yaitu duration.
- Terdapat 3 kolom dengan tipe data int64 yaitu unnamed: 0, days_left, dan price. Kolom price adalah fitur targetnya.
"""

airlines.describe()

"""## Null Check"""

airlines.isna().sum()

"""Dari informasi di atas tidak terdapat nilai yang null.

## Outlier Check
"""

airlines.plot(kind='box', subplots=True, layout=(3,3), figsize=(18,15))
plt.show()

"""Terdapat outlier pada durasi, sehingga kita perlu menghapus sebagian dari outlier tersebut."""

Q1 = airlines.quantile(0.25)
Q3 = airlines.quantile(0.75)
IQR=Q3-Q1
airlines=airlines[~((airlines<(Q1-1.5*IQR))|(airlines>(Q3+1.5*IQR))).any(axis=1)]

# Cek ukuran dataset setelah kita drop outliers
airlines.shape

"""Data yang bersih ada 297.920 sampel.

## Univariate Analysis

### Data Kategorikal

#### 1. Fitur airline
"""

count = airlines['airline'].value_counts()
percent = 100*airlines['airline'].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)

count.plot(kind='bar', title='airline')

"""> Terdapat 6 Maskapai yang berbeda dan Vistara yang menjadi maskapai paling banyak digunakan.

#### 2. Fitur flight
"""

count = airlines['flight'].value_counts()
percent = 100*airlines['flight'].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)

count[:10].plot(kind='bar', title='flight')

num_of_flight = len(airlines['flight'].value_counts())
print(f'Num of flight: {num_of_flight}')

"""> Terdapat 1561 Penerbangan dan UK-706 menjadi penerbangan yang paling sibuk.

#### 3. Fitur source_city
"""

count = airlines['source_city'].value_counts()
percent = 100*airlines['source_city'].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)

count.plot(kind='bar', title='source_city')

"""> Terdapat 6 kota asal dan penerbangan dari Kota Delhi menjadi paling banyak.

#### 4. Fitur destination_city
"""

count = airlines['destination_city'].value_counts()
percent = 100*airlines['destination_city'].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)

count.plot(kind='bar', title='destination_city')

"""> Terdapat 6 kota tujuan dan penerbangan menuju Kota Mumbai menjadi yang paling banyak.

#### 5. Fitur departure_time
"""

count = airlines['departure_time'].value_counts()
percent = 100*airlines['departure_time'].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)

count.plot(kind='bar', title='departure_time')

"""> Waktu numerik diubah menjadi 6 kategorikal, dengan penerbangan di waktu pagi menjadi yang paling sibuk.

#### 6. Fitur arrival_time
"""

count = airlines['arrival_time'].value_counts()
percent = 100*airlines['arrival_time'].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)

count.plot(kind='bar', title='arrival_time')

"""> Waktu numerik diubah menjadi 6 kategorikal, dengan kedatangan di waktu malam menjadi yang paling banyak.

#### 7. Fitur stops
"""

count = airlines['stops'].value_counts()
percent = 100*airlines['stops'].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)

count.plot(kind='bar', title='stops')

"""> Jumlah transit penerbangan diubah menjadi 3 kategorikal dan penerbangan sekali menjadi yang paling banyak.

#### 8. Fitur class
"""

count = airlines['class'].value_counts()
percent = 100*airlines['class'].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)

count.plot(kind='bar', title='class')

"""> Maskapai dengan kelas Ekonomi menjadi yang paling banyak digunakan dengan 68%.

### Data Numerikal
"""

airlines.hist(bins=50, figsize=(20,15))
plt.show()

"""Dari histogram "price", dapat diperoleh beberapa informasi antara lain:

- Peningkatan harga tiket pesawat sebanding dengan penurunan jumlah sampel. Hal ini dengan jelas dari histogram "price" yang mengalami penurunan seiring dengan semakin banyaknya jumlah sampel (sumbu y).
- Rentang harga tiket cukup tinggi dari skala puluhan ribu dolar hingga sekitar \$90000.
- Setengah harga tiket pesawat bernilai di bawah \$10000.
- Distribusi harga miring ke kanan (right-skewed). Hal ini akan berimplikasi pada model.

## Multivariate Analysis

#### 7 Kota asal dan tujuan berdasarkan harga
"""

source_and_dest = airlines.groupby(['source_city', 'destination_city'])['price'].sum().reset_index().sort_values(["price"], ascending=False)
source_and_dest[0:7]

"""#### 5 Penerbangan dan Maskapai berdasarkan harga"""

airline_and_flight = airlines.groupby(['flight', 'airline'])["price"].count().reset_index().sort_values("price", ascending=False)
airline_and_flight[0:5]

plt.figure(figsize=(12, 8))
colors = sns.color_palette('Set2', len(airline_and_flight[0:5]))
ax = sns.barplot(x='flight', y='price', hue='airline', data=airline_and_flight[0:5], palette=colors)
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=10, color='black')

plt.xlabel('Penerbangan')
plt.ylabel('Jumlah harga')
plt.title('5 Penerbangan dan Maskapai Teratas Berdasarkan Jumlah Harga')
plt.show()

"""> ℹ **Informasi**: Penerbangan yang paling sering digunakan adalah UK-706 dan maskapai yang digunakan adalah Vistara airlines.

#### **Analisa Penerbangan Kelas Ekonomi**
"""

eco = airlines[airlines['class']=='Economy']

eco['airline'].value_counts()

colors = sns.color_palette('pastel')
eco['airline'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=colors)
plt.title('Distribusi maskapai paling digunakan pada kelas ekonomi')

eco_price_per_airline = eco.groupby(['airline'])['price'].sum().reset_index().sort_values(by='price', ascending=False)
eco_price_per_airline

plt.figure(figsize=(8, 5))
colors = sns.color_palette('bright', len(eco_price_per_airline))
ax = sns.barplot(x='airline', y='price', data=eco_price_per_airline, palette=colors)

for p in ax.patches:
    ax.annotate(f'${p.get_height():,.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=10, color='black')

plt.xlabel('Maskapai')
plt.ylabel('Total Harga')
plt.title('Total Harga setiap Maskapai di Kelas Ekonomi')

plt.show()

"""Kota mana saja yang paling banyak menggunakan tiket kelas ekonomi?"""

eco['source_city'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=colors)
plt.title('Distribusi kota asal di kelas ekonomi')

"""> 1 - Maskapai yang paling sering digunakan juga mencapai profit tertinggi adalah Vistara dan setelahnya ada Air India.


> 2 - Kota yang paling banyak menggunakan kelas ekonomi adalah Delhi dan setelahnya adalah Mumbai.

**Analisa Penerbangan di Kelas Bisnis**
"""

bus = airlines[airlines['class']=='Business']

colors = sns.color_palette('deep')
bus['airline'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=colors)
plt.title('Maskapai dengan kelas bisnis yang digunakan')

bus_price_per_airline = bus.groupby(['airline'])['price'].sum().reset_index().sort_values(by='price', ascending=False)
bus_price_per_airline

colors_count = bus['airline'].value_counts()

plt.figure(figsize=(12, 8))
colors = sns.color_palette('Set3', len(colors_count))
ax = sns.barplot(x=colors_count.index, y=colors_count.values, palette=colors)

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='baseline', fontsize=10, color='black')

plt.xlabel('Maskapai')
plt.ylabel('Harga')
plt.title('Distribusi Maskapai di Kelas Bisinis')

plt.show()

bus.groupby(['airline'])['price'].mean().reset_index().sort_values(by='price', ascending=False)

colors = sns.color_palette('bright')
bus['source_city'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=colors)
plt.title('Distribusi kota yang menggunakan kelas bisnis')

"""Apakah harga berbeda pada maskapai?"""

df = airlines.groupby(['airline'])['price'].median()
df

dm = airlines['airline'].value_counts()
total_customers = len(airlines)

percentages = dm / total_customers * 100

# Choose a different color palette from Seaborn
colors = sns.color_palette('Set3', n_colors=len(dm))

plt.figure(figsize=(8, 6))
sns.barplot(x=dm.index, y=dm, palette=colors)

for i, p in enumerate(plt.gca().patches):
    percentage = '{:.1f}%'.format(percentages[i])
    x = p.get_x() + p.get_width() / 2
    y = p.get_height() + 0.5
    plt.text(x, y, percentage, ha='center', va='bottom', fontsize=10, color='black')

plt.title('price vary with Airlines')
plt.xlabel('airlines')
plt.ylabel('prices')

plt.show()

"""Apakah harga tiket berbeda antara Kelas Ekonomi dengan Kelas Bisnis?"""

airlines.groupby(['class'])['price'].mean().plot(kind='pie', title='Perbedaan tiap kelas tiket', autopct='%1.1f%%')

"""Apakah perubahan harga tiket berdasarkan pada waktu keberangkatan dan kedatangan?"""

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=airlines, x='departure_time', y='price', showfliers=False).set_title('harga maskapai berdasarkan waktu keberangkatan', fontsize=15)
plt.subplot(1,2,2)
sns.boxplot(data=airlines, x='arrival_time', y='price', showfliers=False).set_title('harga maskapai berdasarkan waktu kedatangan', fontsize=15)

"""Bagaimana perubahan harga dengan perubahan kota asal dan kota tujuan?"""

source_and_dest_price = sns.relplot(col='source_city', y='price', kind='line', x='destination_city', data=airlines, col_wrap=3)
source_and_dest_price.fig.subplots_adjust(top=0.9)
source_and_dest_price.fig.suptitle('Perubahan Harga dengan Kota asal dan tujuan')

"""> Berdasarkan grafik ini, terlihat bahwa penerbangan yang berangkat dan tiba di Delhi umumnya lebih murah, sedangkan penerbangan yang berangkat dan tiba di Bangalore harganya lebih mahal.

#### **Bagaimana perubahan harga tiket antara kelas ekonomi dengan kelas bisnis?**
"""

fig, axs = plt.subplots (1, 2, gridspec_kw={'width_ratios': [5, 3]}, figsize=(25, 5))
sns.violinplot(x='airline', y='price', data=airlines.loc[airlines['class']=='Economy'].sort_values(by='price', ascending=False), ax=axs[0])
axs[0].set_title('Maskapai berdasarkan kelas ekonomi', fontsize=20)
sns.violinplot(x='airline', y='price', data=airlines.loc[airlines['class']=='Business'].sort_values(by='price', ascending=False), ax=axs[1])
axs[1].set_title('Maskapai berdasarkan kelas bisnis', fontsize=20)

"""> Harga tiket pesawat bervariasi tergantung maskapainya. Di antara maskapai yang ada, Air India dan Vistara tergolong yang termahal, sedangkan AirAsia menawarkan harga yang paling bersahabat. Khusus untuk kelas bisnis, Vistara mematok harga tertinggi dibanding maskapai lain, termasuk AirAsia.

#### **Apakah harga tiket berubah berdasarkan pemberhentian penerbangan?**
"""

airlines['stops'].value_counts().plot(kind='pie', autopct='%1.1f%%', title='Distribusi pemberhentian penerbangan', colors=colors)

sns.barplot(y='price', x='airline', hue='stops', data=airlines.sort_values("price", ascending=False))
plt.title("Distribusi harga tiket berdasarkan jumlah pemberhentian untuk setiap maskapai")

"""> 83.6 persentase penerbangan sekali pemberhentian yang digunakan maskapai dan maskapai yang paling berhenti sekali paling banyak adalah Vistara dan Air India."""

df = airlines.copy()

airlines['source_to_destination'] = airlines['source_city']+' to '+airlines['destination_city']
airlines

airlines.groupby(['source_to_destination', 'airline'])['price'].size().reset_index().sort_values(by='price', ascending=False)

"""Berapa mean harga dan durasi penerbangan dengan kota asal ke kota tujuan?"""

mean_harga = airlines.groupby(['source_to_destination'])['price'].mean().reset_index().sort_values(by='price', ascending=False)
mean_harga[0:5]

"""Jumlah pemberhentian untuk setiap perjalanan"""

colors=sns.color_palette('bright')
plt.clf()
plt.figure(figsize=(18,7))
sns.countplot(x='source_to_destination', hue='stops', data=airlines, palette=colors)
plt.xticks(rotation=90)
plt.tight_layout()
plt.title('Jumlah pemberhentian untuk setiap perjalanan.')
plt.show()

days_left = airlines.groupby(['days_left'])['price'].mean()

fig, axs = plt.subplots(1,2,figsize=(18,5))
plt.tight_layout()
sns.lineplot(x='days_left',y='price', data=airlines, ax=axs[0])
sns.boxplot(x='days_left', y='price', data=airlines, ax=axs[1])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

"""<div style="background-color:rgba(250,230,7,1);">

> ⚠ Informasi: Grafik ini mengilustrasikan terdapat peningkatan harga hingga 20 hari sebelum penerbangan, diikuti dengan penurunan yang sangat tajam menjelang 1 hari, menjadi harga 3 kali lebih murah. Pola ini menyarankan, maskapai boleh mengurangi harga menjelang waktu keberangkatan untuk mengisi kursi kosong dan memastikan kursi penuh di pesawat mereka.

</div>
"""

airlines['days_left'].unique()

def offer(x):
    if x['days_left']==1:
        return ("5% offer")
    elif x['days_left']>=2 and x['days_left']<=8:
        return('7% offer')
    elif x['days_left']>8 and x['days_left']>=15:
        return('10% offer')
    else:
        return('no offer')

airlines['offer']=airlines.apply(offer,axis=1)

airlines['offer'].value_counts().plot(kind='pie',autopct = "%1.1f%%",colors = colors)
plt.title('Penawaran untuk setiap tiket berdasarkan waktu booking')

"""> Berdasarkan analisis, kebanyakan pesawat di hari pertama dibooking (dilihat dari 10% adalah terbanyak), dan penerbangan tidak diisi pada hari dimana terdapat pengurangan harga pesawat, sehingga perlu mengurangi harga lebih lanjut."""

city_count = dict(airlines['source_city'].value_counts())
city_count

def offer_city(count):
    if  count>=60000:
        return('offer 5%')
    if   45000 <= count < 60000:
         return('offer 7%')
    if   30000<= count <45000:
        return('offer 10%')
    else:
        return('no offer')

airlines['offer_city']=airlines['source_city'].map(city_count).apply(offer_city)

airlines

"""# Data Preparation"""

sns.pairplot(df, hue='airline')

duplicated = df.duplicated()
duplicated.sum()

df.describe(include='object')

df['airline'].value_counts()

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df['airline'] = label_encoder.fit_transform(df['airline'])
df['source_city'] = label_encoder.fit_transform(df['source_city'])
df['departure_time'] = label_encoder.fit_transform(df['departure_time'])
df['stops'] = label_encoder.fit_transform(df['stops'])
df['arrival_time'] = label_encoder.fit_transform(df['arrival_time'])
df['destination_city'] = label_encoder.fit_transform(df['destination_city'])
df['class'] = label_encoder.fit_transform(df['class'])

df.drop(['Unnamed: 0', 'flight'], axis=1, inplace=True)

df.head()

correlation_with_price = df.corr()['price'].sort_values(ascending=False)
print(correlation_with_price)

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr().round(2)

# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.figure(figsize=(20,20))
plt.show()

df.mean()

X = df.drop(['price'], axis=1).values
y = df.price.values

"""Standarization"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit_transform(X)
X = pd.DataFrame(scaler)

X.describe().T[['min', 'mean', 'std', '50%', 'max']].style.background_gradient(axis=1)

"""# Modeling"""

# Splitting to training and testing data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

X_train.shape

y_train.shape

X_test.shape

y_test.shape

from sklearn.metrics import r2_score, \
mean_absolute_error, \
mean_squared_error, \
mean_absolute_percentage_error

# analisis model
models_name = ["Linear Regression",
                 "Decision Tree Regressor",
                 "Random Forest Regressor",
                 "Gradient Boosting Regressor",
                 "AdaBoostRegressor",
                 "XGBRegressor"]
eval_matrices = ['MAE', 'MSE', 'MAPR', 'R2 Squared']
models = pd.DataFrame(index=eval_matrices,
		      columns=models_name)

"""### Linear Regression"""

from sklearn import linear_model
LR = linear_model.LinearRegression()

LR.fit(X_train, y_train)
print("train score: {:.2f}".format(LR.score(X_train,y_train)))
print("test score: {:.2f}".format(LR.score(X_train,y_train)))
y_pred = LR.predict(X_test)
LR_pred = pd.DataFrame({"y_test":y_test,'y_pred':y_pred})

models.loc['Linear Regression', 'R2'] = r2_score(y_test, y_pred)
print("r2 score: {:.2f}".format(models.loc['Linear Regression', 'R2']))
models.loc['Linear Regression', 'MAE'] = mean_absolute_error(y_test,y_pred)
print("MAE: {:.2f}".format(models.loc['Linear Regression', 'MAE']))
models.loc['Linear Regression', 'MSE'] = mean_squared_error(y_test,y_pred)
print("MSE: {:.2f}".format(models.loc['Linear Regression', 'MSE']))
models.loc['Linear Regression', 'MAPR'] = mean_absolute_percentage_error(y_test,y_pred)
print("MAPR: {:.2f}".format(models.loc['Linear Regression', 'MAPR']))

LR.intercept_

LR.coef_

"""### Decision Tree"""

from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

Rtree = DecisionTreeRegressor(criterion='friedman_mse', max_depth=10, max_features=9, random_state=42)

Rtree.fit(X_train, y_train)
y_pred = Rtree.predict(X_test)
Rtree_pred = pd.DataFrame({"y_test":y_test,'y_pred':y_pred})
print("train score: {:.2f}".format(Rtree.score(X_train,y_train)))
print("test score: {:.2f}".format(Rtree.score(X_train,y_train)))

models.loc['Decision Tree Regressor', 'R2'] = r2_score(y_test, y_pred)
print("r2 score: {:.2f}".format(models.loc['Decision Tree Regressor', 'R2']))
models.loc['Decision Tree Regressor', 'MAE'] = mean_absolute_error(y_test,y_pred)
print("MAE: {:.2f}".format(models.loc['Decision Tree Regressor', 'MAE']))
models.loc['Decision Tree Regressor', 'MSE'] = mean_squared_error(y_test,y_pred)
print("MSE: {:.2f}".format(models.loc['Decision Tree Regressor', 'MSE']))
models.loc['Decision Tree Regressor', 'MAPR'] = mean_absolute_percentage_error(y_test,y_pred)
print("MAPR: {:.2f}".format(models.loc['Decision Tree Regressor', 'MAPR']))

"""### Random Forest"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error

rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, max_features=9)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
Rforest_pred = pd.DataFrame({"y_test":y_test,'y_pred':y_pred})
print("train score: {:.2f}".format(rf_model.score(X_train,y_train)))
print("test score: {:.2f}".format(rf_model.score(X_train,y_train)))

models.loc['Random Forest Regressor', 'R2'] = r2_score(y_test, y_pred)
print("r2 score: {:.2f}".format(models.loc['Random Forest Regressor', 'R2']))
models.loc['Random Forest Regressor', 'MAE'] = mean_absolute_error(y_test,y_pred)
print("MAE: {:.2f}".format(models.loc['Random Forest Regressor', 'MAE']))
models.loc['Random Forest Regressor', 'MSE'] = mean_squared_error(y_test,y_pred)
print("MSE: {:.2f}".format(models.loc['Random Forest Regressor', 'MSE']))
models.loc['Random Forest Regressor', 'MAPR'] = mean_absolute_percentage_error(y_test,y_pred)
print("MAPR: {:.2f}".format(models.loc['Random Forest Regressor', 'MAPR']))

"""### Gradient Boosting"""

from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
y_pred = gb_model.predict(X_test)
gb_pred = pd.DataFrame({"y_test":y_test,'y_pred':y_pred})
print("train score: {:.2f}".format(gb_model.score(X_train,y_train)))
print("test score: {:.2f}".format(gb_model.score(X_train,y_train)))

models.loc['Gradient Boosting Regressor', 'R2'] = r2_score(y_test, y_pred)
print("r2 score: {:.2f}".format(models.loc['Gradient Boosting Regressor', 'R2']))
models.loc['Gradient Boosting Regressor', 'MAE'] = mean_absolute_error(y_test,y_pred)
print("MAE: {:.2f}".format(models.loc['Gradient Boosting Regressor', 'MAE']))
models.loc['Gradient Boosting Regressor', 'MSE'] = mean_squared_error(y_test,y_pred)
print("MSE: {:.2f}".format(models.loc['Gradient Boosting Regressor', 'MSE']))
models.loc['Gradient Boosting Regressor', 'MAPR'] = mean_absolute_percentage_error(y_test,y_pred)
print("MAPR: {:.2f}".format(models.loc['Gradient Boosting Regressor', 'MAPR']))

"""### Ada Boost"""

from sklearn.ensemble import AdaBoostRegressor

ada_model = AdaBoostRegressor(random_state=42)
ada_model.fit(X_train, y_train)
y_pred = ada_model.predict(X_test)
ada_pred = pd.DataFrame({"y_test":y_test,'y_pred':y_pred})
print("train score: {:.2f}".format(ada_model.score(X_train,y_train)))
print("test score: {:.2f}".format(ada_model.score(X_train,y_train)))

models.loc['AdaBoostRegressor', 'R2'] = r2_score(y_test, y_pred)
print("r2 score: {:.2f}".format(models.loc['AdaBoostRegressor', 'R2']))
models.loc['AdaBoostRegressor', 'MAE'] = mean_absolute_error(y_test,y_pred)
print("MAE: {:.2f}".format(models.loc['AdaBoostRegressor', 'MAE']))
models.loc['AdaBoostRegressor', 'MSE'] = mean_squared_error(y_test,y_pred)
print("MSE: {:.2f}".format(models.loc['AdaBoostRegressor', 'MSE']))
models.loc['AdaBoostRegressor', 'MAPR'] = mean_absolute_percentage_error(y_test,y_pred)
print("MAPR: {:.2f}".format(models.loc['AdaBoostRegressor', 'MAPR']))

"""### XG Boost"""

from xgboost import XGBRegressor
xgb = XGBRegressor()
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
xgb_pred = pd.DataFrame({"y_test":y_test,'y_pred':y_pred})
print("train score: {:.2f}".format(xgb.score(X_train,y_train)))
print("test score: {:.2f}".format(xgb.score(X_train,y_train)))

models.loc['XGBRegressor', 'R2'] = r2_score(y_test, y_pred)
print("r2 score: {:.2f}".format(models.loc['XGBRegressor', 'R2']))
models.loc['XGBRegressor', 'MAE'] = mean_absolute_error(y_test,y_pred)
print("MAE: {:.2f}".format(models.loc['XGBRegressor', 'MAE']))
models.loc['XGBRegressor', 'MSE'] = mean_squared_error(y_test,y_pred)
print("MSE: {:.2f}".format(models.loc['XGBRegressor', 'MSE']))
models.loc['XGBRegressor', 'MAPR'] = mean_absolute_percentage_error(y_test,y_pred)
print("MAPR: {:.2f}".format(models.loc['XGBRegressor', 'MAPR']))

"""# Evaluation"""

eval = pd.DataFrame(columns=eval_matrices,
                    index = models_name)

eval.index.name = 'Model'

model_dict = {models_name[0] : LR,
              models_name[1] : Rtree,
              models_name[2] : rf_model,
              models_name[3] : gb_model,
              models_name[4] : ada_model,
              models_name[5] : xgb}

for name, model in model_dict.items():
    eval.loc[name, eval_matrices[3]] = r2_score(y_test, model.predict(X_test))
    eval.loc[name, eval_matrices[0]] = mean_absolute_error(y_test, model.predict(X_test))
    eval.loc[name, eval_matrices[1]] = mean_squared_error(y_test, model.predict(X_test))
    eval.loc[name, eval_matrices[2]] = mean_absolute_percentage_error(y_test, model.predict(X_test))

# Function to highlight minimum MSE, RMSE, MAE, and maximum R-squared values and make the font bold
def highlight_min_max(val):
    style = ''
    if isinstance(val, (int, float)):
        if val == eval['MSE'].min():
            style += 'background-color: rgba(0, 128, 0, 0.3); color: black; font-weight: bold;'
        if val == eval['MAPR'].min():
            style += 'background-color: rgba(0, 128, 0, 0.3); color: black; font-weight: bold;'
        if val == eval['MAE'].min():
            style += 'background-color: rgba(0, 128, 0, 0.3); color: black; font-weight: bold;'
        if val == eval['R2 Squared'].max():
            style += 'background-color: rgba(0, 0, 128, 0.3); color: white; font-weight: bold;'
    return style

styled_eval = eval.style.applymap(highlight_min_max)
styled_eval

train_eval = pd.DataFrame(columns=eval_matrices,
                    index = models_name)

train_eval.index.name = 'Model'

model_dict = {models_name[0] : LR,
              models_name[1] : Rtree,
              models_name[2] : rf_model,
              models_name[3] : gb_model,
              models_name[4] : ada_model,
              models_name[5] : xgb}

for name, model in model_dict.items():
    train_eval.loc[name, eval_matrices[3]] = r2_score(y_train, model.predict(X_train))
    train_eval.loc[name, eval_matrices[0]] = mean_absolute_error(y_train, model.predict(X_train))
    train_eval.loc[name, eval_matrices[1]] = mean_squared_error(y_train, model.predict(X_train))
    train_eval.loc[name, eval_matrices[2]] = mean_absolute_percentage_error(y_train, model.predict(X_train))

train_eval

"""## Visualisasi Metrik Evaluasi"""

# MAE Visualization
mae = pd.DataFrame(columns=['train', 'test'],
                    index = models_name)

mae['train'] = train_eval['MAE']
mae['test'] = eval['MAE']

# MSE Visualization
mse = pd.DataFrame(columns=['train', 'test'],
                    index = models_name)

mse['train'] = train_eval['MSE']
mse['test'] = eval['MSE']

# MAPR Visualization
mapr = pd.DataFrame(columns=['train', 'test'],
                    index = models_name)

mapr['train'] = train_eval['MAPR']
mapr['test'] = eval['MAPR']

# R Squared Visualization
r2 = pd.DataFrame(columns=['train', 'test'],
                    index = models_name)

r2['train'] = train_eval['R2 Squared']
r2['test'] = eval['R2 Squared']

fig, ax = plt.subplots(2,2, figsize=(12,10))

fig.tight_layout(h_pad=2, w_pad=15)
ax[0,0].grid(zorder=0)
mae.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax[0,0], zorder=3)
ax[0,0].set_title('MAE Evaluations')

mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax[0,1], zorder=3)
ax[0,1].grid(zorder=0)
ax[0,1].set_title('MSE Evaluations')

mapr.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax[1,0], zorder=3)
ax[1,0].grid(zorder=0)
ax[1,0].set_title('MAPR Evaluations')

r2.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax[1,1], zorder=3)
ax[1,1].grid(zorder=0)
ax[1,1].set_title('R2 Squared Evaluations')

plt.show()

"""## Prediction Testing"""

prediksi = X_test.iloc[0:50].copy()
pred_dict = {'y_true':y_test[0:50]}
for name, model in model_dict.items():
    pred_dict[name] = model.predict(prediksi).round(2)

pred_df = pd.DataFrame(pred_dict)

pred_df.head()

fig, axs = plt.subplots(3,2, figsize=(10,10))
fig.suptitle('Prediksi masing-masing model')
axs[0,0].set_title('Linear Regression')
axs[0,0].plot(pred_df[['y_true','Linear Regression']])
axs[0,1].set_title('Decision Tree Regressor')
axs[0,1].plot(pred_df[['y_true','Decision Tree Regressor']])
axs[1,0].set_title('Random Forest Regressor')
axs[1,0].plot(pred_df[['y_true','Random Forest Regressor']])
axs[1,1].set_title('Gradient Boosting Regressor')
axs[1,1].plot(pred_df[['y_true','Gradient Boosting Regressor']])
axs[2,0].set_title('AdaBoostRegressor')
axs[2,0].plot(pred_df[['y_true','AdaBoostRegressor']])
axs[2,1].set_title('XGBRegressor')
axs[2,1].plot(pred_df[['y_true','XGBRegressor']])
plt.show()