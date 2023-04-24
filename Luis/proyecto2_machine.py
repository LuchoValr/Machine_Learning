import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_1 = pd.read_table('https://raw.githubusercontent.com/LuchoValr/Machine_Learning/main/Luis/Data1.txt', 
                        encoding = "utf-16", sep = ',')

data_2 = pd.read_table('https://raw.githubusercontent.com/LuchoValr/Machine_Learning/main/Luis/Data2.txt',
                       encoding = "utf-16", sep = ',')

#El primer dataset se estima con una regresion lineal, ya que la dependiente es una variable continua
data_1.head()
#El segundo dataset se estima con una regresion logistica, ya que la dependiente es una variable discreta o binaria
data_2.head()

#El segundo dataset la dependiente no está dada de manera númerica
data_2['SAFE'] = data_2['SAFE'].replace({'NO' : 0, 'YES' : 1})
data_2.head()

#Estimaciones con la libreria sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
#Regresion primer dataset
X_data_1 = np.array(data_1['BACKYARD SIZE(m2)']).reshape((-1, 1))
Y_data_1 = np.array(data_1['PRICE(USD)'])

model_1 = LinearRegression(fit_intercept = False)
model_1.fit(X_data_1, Y_data_1)

print(f'intercepto (b): {model_1.intercept_}''\n', 
      f'pendiente (w): {model_1.coef_}')

#Regresion segundo dataset
X_data_2 = np.array(data_2['Security Level']).reshape((-1, 1))
Y_data_2 = np.array(data_2['SAFE'])

model_2 = LogisticRegression()
model_2.fit(X_data_2, Y_data_2)

print(f' la pendiente (w) es {model_2.coef_}''\n', 
      f'el intercepto (b) es {model_2.intercept_}')

#Procedimiento paso a paso primer dataset (regresión lineal)

sns.scatterplot(data = data_1, x = data_1['BACKYARD SIZE(m2)'], y = data_1['PRICE(USD)'])
plt.show()

w = 16000
b = 0
x = np.linspace(0, data_1['BACKYARD SIZE(m2)'].max(), 100)
y = (w*x)+b

data_1.plot.scatter(x = 'BACKYARD SIZE(m2)', y = 'PRICE(USD)')
plt.plot(x, y, '-r')
plt.ylim(0, data_1['PRICE(USD)'].max()*1.1)
plt.show()

#Prediccion

data_1['Pred'] = data_1['BACKYARD SIZE(m2)']*w+b

#Funcion de error

data_1['Diff'] = data_1['Pred'] - data_1['PRICE(USD)']
data_1['Cuad'] = data_1['Diff']**2
data_1.head()

w = np.linspace(1000, 28000, 50)
grid_error = pd.DataFrame(w, columns = ['w'])
grid_error.head()

def sum_error(w, train):
  b = 0
  data_1['Pred'] = data_1['BACKYARD SIZE(m2)']*w+b
  data_1['Diff'] = data_1['Pred'] - data_1['PRICE(USD)']
  data_1['Cuad'] = data_1['Diff']**2
  data_1.head()
  return(data_1['Cuad'].mean())

grid_error['error'] = grid_error['w'].apply(lambda x: sum_error(x, train = data_1))
grid_error.head()

grid_error.plot(x = 'w', y = 'error')
plt.show()

#Procedimiento paso a paso regresión logistica para el segundo dataset

data_2.plot.scatter(x = 'Security Level', y = 'SAFE', c = 'b')
plt.show()

#Parametros al ojo
w = 7.22 
b = -39.81

x = np.linspace(0, data_2['Security Level'].max(), 100) #Inicio, final, cantidad
y = 1/(1 + np.exp(-(w*x+b)))

data_2.plot.scatter(x = 'Security Level', y = 'SAFE', c = 'b')
plt.plot(x, y, '-r')
plt.ylim(0, data_2['SAFE'].max()*1.1)
#plt.grid()
plt.show()

array = np.mgrid[3.22:9.22:0.1, -40.81:-30.81:5].reshape(2,-1).T
df = pd.DataFrame(data = array, columns = ['w', 'b'])
df['w'] = np.round(df['w'], 6)
df['b'] = np.round(df['b'], 6)

def sum_error_df(df):
  data_2['sigmoid'] = 1/(1 + np.exp(-(data_2['Security Level']*df['w']+df['b'])))
  data_2['loss_xi'] = -data_2['SAFE']*np.log(data_2['sigmoid'])-(1 - data_2['SAFE'])*np.log(1 - data_2['sigmoid'])
  j_cost = data_2['loss_xi'].mean()
  return(j_cost)

sum_error_df(df)

df['error'] = df.apply(sum_error_df, axis = 1)
df.sort_values(by = ['error']).head()

df_3d = df.pivot(index = 'w', columns = 'b', values = 'error')
df_3d.head()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors

x = df_3d.columns
y = df_3d.index
X, Y = np.meshgrid(x, y)
Z = df_3d
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_surface(X, Y, Z, color = 'r')
plt.show()

plt.contourf(Y, X, Z, alpha = 0.7, cmap = plt.cm.jet)
plt.show()

def delta_j_w(w, b):
   data_2['sigmoid'] = 1/(1 + np.exp(-(data_2['Security Level']*df['w']+df['b'])))
   data_2['partial_loss'] = (data_2['sigmoid'] - data_2['SAFE'])*data_2['Security Level']
   derivate = data_2['partial_loss'].mean()
   return(derivate)

def delta_j_b(w, b):
  data_2['sigmoid'] = 1/(1 + np.exp(-(data_2['Security Level']*w+b)))
  data_2['partial_loss'] = (data_2['sigmoid'] - data_2['SAFE'])
  derivate = data_2['partial_loss'].mean()
  return(derivate)

w_0 = 7.22 
b_0 = -39.81
delta_j_w(w_0, b_0)
delta_j_b(w_0, b_0)

alpha_w = 0.001
alpha_b = 0.1
w_new = w_0 - alpha_w * delta_j_w(w_0, b_0)
b_new = b_0 - alpha_b * delta_j_b(w_0, b_0)
w_0 = w_new
b_0 = b_new
w_0 = np.round(w_0, 5)
b_0 = np.round(b_0, 5)
print(w_0, b_0)





     