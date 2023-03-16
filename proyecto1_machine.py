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
    

