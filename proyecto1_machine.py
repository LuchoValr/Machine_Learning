import pandas as pd 
import matplotlib.pyplot as plt

#Lectura base de datos

data = pd.read_table('https://raw.githubusercontent.com/LuchoValr/Pruebas-codigos/main/aerolinea.txt', 
                   header = None, sep = ',', encoding="utf-16")
data.head()

data.shape
data.size

#Nombrando columnas

data = data.rename(columns = {0:'sexo', 1:'clase', 2:'destino', 3:'temporada', 4:'edad',
                       5:'compra', 6:'equipaje'})
data.head()

#Recorrido por cada variable

for col in data:
    print(data[col].value_counts(), '\n''-----------------------')

#Graficos de barras

for col in data:
    bar = data[col].value_counts()
    bar.plot(kind = 'bar', color = ['orange', 'red'])
    plt.xlabel(col)
    plt.ylabel(f'Número de vuelos por {col}')
    plt.title(f'Registro de vuelos por {col}')
    plt.show()

#Graficos de torta

for col in data:
    pie = data[col].value_counts()
    pie.plot(kind = 'pie', colors = ['orange', 'red'],
             shadow = True, autopct = '%.2f%%')
    plt.title(f'Registro de vuelos por {col}', fontsize = 14)
    plt.legend(loc = 'best')
    plt.show()



    

