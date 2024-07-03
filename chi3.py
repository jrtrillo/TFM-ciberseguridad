import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
#Variables fijas
columna_variable = 7
chi = 3

## Leer los documentos
file = '2tra.csv'
data = pd.read_csv(file,sep=',',header=None).values
train = data[:, 0:-1]
y_train = data[:,-1].astype(int)

file2 = '2tst.csv'
data = pd.read_csv(file2,sep=',',header=None).values
dataset_test = data[:, 0:-1]
y_test = data[:,-1].astype(int)

print("etiquetas train")
print(y_train)
print("etiquetas test")
print(y_test)

## Número de variables diferentes que existen.
label = []

for i in range(len(y_train)):

	if y_train[i] not in label:
		label.append(y_train[i])

def valor_pertenencia(valor,columna,c):
	
	i = columna * chi + c

	if(m_p[i][0] <= valor and valor <= m_p[i][1]):
		
		if (m_p[i][0] == valor and m_p[i][1] == valor):
		
			return 1.0
			
		else:
			
			return (valor - m_p[i][0])/(m_p[i][1] - m_p[i][0])
			
	elif(m_p[i][1] <= valor and valor <= m_p[i][2]):
		
		if(m_p[i][2] == valor and m_p[i][1] == valor):
			
			return 1.0
			
		else:
			
			return (m_p[i][2]- valor)/(m_p[i][2] - m_p[i][1])
			
	else:
			
		return 0
			


def matriz_pertenencia(m_maxmin):
	filas = (chi*(columna_variable))
	index = 0
	ind = 0
	m_p = np.zeros((filas, 5))

	for i in range(columna_variable):
		for j in range(chi):
			base = (m_maxmin[i][1] - m_maxmin[i][0]) / (chi - 1)
			filamp = chi*i+j 
			if((j%chi) == 0 ):
				m_p[filamp][0] =  m_maxmin[i][0]
				m_p[filamp][1] =  m_maxmin[i][0]
				m_p[filamp][2] =  m_maxmin[i][0] + base
			elif((j%chi) == (chi -1)):
				m_p[filamp][0] = m_p[filamp-1][1]
				m_p[filamp][1] =  m_maxmin[i][1]
				m_p[filamp][2] =  m_maxmin[i][1]
			else:
				m_p[filamp][0] = m_p[filamp-1][1]
				m_p[filamp][1] = m_p[filamp-1][2]
				m_p[filamp][2] = m_p[filamp-1][2] + base 

		
			m_p[filamp][3] = i
			m_p[filamp][4] = j

	return m_p

def valor_maximo(valor,columna):
	#
	# Explicacion:
	#
	#	Calcula todos los valores de pertenencia de una columna para un valor concreto.
	#
	# Argumento:
	#
	#		* Valor: Tipo de dato Float. Es el valor asociado a la matriz de train/test.
	#		* Columna: Tipo de dato Int. Es la columna asociada al valor.
	#
	# Return:
	#
	#	Devuelve el valor maximo del valor de pertenencia, la clase asociada al valor de pertenencia 
	#	y la columna perteneciente ese valor maximo.
	#
	resultado = []
	valor_max = -50
	clase = -50
	for i in range(chi):
		v_resultados = valor_pertenencia(valor,columna,i)
		if(v_resultados > valor_max):
			valor_max = v_resultados
			clase = i

	
	resultado.append(clase)
	resultado.append(valor_max)
	
	return resultado

def vec_in_vec(vector,v_matriz): 
	#
	# Explicación:
	#
	#	Funcion booleana que permite comparar dos vectores y saber si son iguales o no.
	#
	# Argumentos:
	#
	#		* vector: Vector de tipo int/float. Es un vector de números enteros o reales.
	#		* v_matriz: Vector de tipo int/float. Es un vector de números enteros o reales. 
	#
	# Return:
	#
	#	Devuelve True si los dos vectores son iguales y False si son distintos.
	#
	contador = 0
	for i in range(columna_variable):
		if(vector[i] == v_matriz[i]):
			contador = contador + 1
			if(contador == columna_variable):
				return True
	return False

def pesos_candidatos(matriz_train,vec_re,etiquetas):
	#
	# Explicacion:
	#
	#	Calcula el peso de cada clase para cada regla asociada.
	#
	# Argumentos:
	#
	#		*vector: Vector tipo int. Es el vector con los antecedentes del cual se desea conocer 
	#				 el peso que se tiene para cada clase.
	#		*matriz: Matriz tipo float. Esta matriz contiene todas las reglas y pesos de todos los ejemplos.
	#
	# Return:
	#
	#	Devuelve el vector cuya longitud es el numero de clases diferentes y 
	#   contiene el peso de cada clase.
	#
	suma = []
	for k in range(len(label)):
		suma.append(0)
	

	for i in range(len(matriz_train)):
		rmax = 1

		for j in range(columna_variable):
			
			r = valor_pertenencia(matriz_train[i][j],j,int(vec_re[j]))
			
			
			rmax = rmax * r


		suma[int(etiquetas[i])] = suma[int(etiquetas[i])] + rmax 

	print(suma)
	return suma

def peso_regla(pesos_candidatos):
	#
	# Explicacion:
	#
	#	Calcula el peso maximo de la regla.
	#
	# Argumentos:
	#
	#		*pesos_candidatos: Vector tipo float. Es el vector de pesos de cada clase de una regla.
	#
	# Return:
	#
	#	Devuelve el peso maximo de la regla y la clase que tiene asociada.
	#
	maximo = -1
	total = 0
	clase = -1
	for i in range(len(pesos_candidatos)):
		total = pesos_candidatos[i] + total
	for i in range(len(pesos_candidatos)):
		if((pesos_candidatos[i]/total) > maximo):
			maximo = pesos_candidatos[i]/total
			clase = i
	resultado = []
	resultado.append(maximo)
	resultado.append(clase)
	return resultado

def test(vector,matriz):
	#
	# Explicacion:
	#
	#	Permite evaluar cualquier fila conociendo su clase y su peso asociado.
	#
	# Argumentos:
	#
	#		*vector: Vector tipo float. Es la fila que se desea evaluar.
	#		*matriz: Matriz tipo float. Esta matriz de reglas con sus pesos asociados.
	#
	# Return:
	#
	#	Devuelve la clase asociada y su peso.
	#
	#vector_clase = []

#	for i in range(columna_variable):
#		clase = valor_maximo(vector[i],i)
#		vector_clase.append(clase[0])



#	for i in range(len(matriz)):
#		if(vec_in_vec(vector_clase,matriz[i])):

#			return matriz[i][columna_variable+1]


#	return 0
	cl = -50
	val = -50
	for i in range(len(matriz)):

		res = 50
		for j in range(columna_variable):
			candidato = valor_pertenencia(vector[j],j,int(matriz[i][j]))
			if(candidato <= res):
				res = candidato

		res = res * matriz[i][columna_variable]
		
		if(val < res ):
			val = res
			cl = matriz[i][columna_variable+1]


	return cl

#


max_min = np.zeros((columna_variable, 3))
for i in range(columna_variable):
	max_min[i][0] = 100000


for j in range(columna_variable):
	for i in range(len(train)):
	
				if(max_min[j][0] > train[i][j]):
					max_min[j][0] = train[i][j]
				if(max_min[j][1] < train[i][j]):
					max_min[j][1] = train[i][j]
				
	max_min[j][2]=j



#print(max_min)
m_p = matriz_pertenencia(max_min)
print("Matriz de pertenencia")
print(m_p)
matrix_pertenencia = np.zeros((len(train), columna_variable))

# De aqui para abajo arreglar ahora mismo generamos bien las reglas pero no generamos bien los pesos

matriz_reglas = np.zeros((len(train), columna_variable+2))

for i in range(len(train)):
	matriz_reglas[i][0] = 7

for i in range(len(train)):
	for j in range(columna_variable):

		vector_asociado = valor_maximo(train[i][j],j)
		matrix_pertenencia[i][j]= vector_asociado[0]

print(" ")
print("Matriz de ejemplos")
print(matrix_pertenencia)
print(len(matrix_pertenencia))


#
# Calcula la matriz de las reglas. Sus columnas son:
#					* Cada una de las columnas del conjunto train.
#					* La clase asociada.
#					* La valor asociado a la regla.
# El numero de filas son  cada uno de ejemplos del conjunto train (aunque no se utilizaran todas las filas).
#

indice = 0

for i in range(len(train)):
	contador = 0
	for j in range(len(train)):
		if(vec_in_vec(matriz_reglas[j],matrix_pertenencia[i]) == False):
			contador = contador + 1
			if(contador == len(train)):
				for k in range(columna_variable):
					matriz_reglas[indice][k] = (matrix_pertenencia[i][k])
				indice = indice + 1

	
#
# Elimina las filas no son utilizadas. Sus columnas son:
#					* Cada una de las columnas del conjunto train.
#					* La clase asociada.
#					* La valor asociado a la regla.
# El numero de filas es la variable tipo int llamada indice.
#
matriz_final = np.zeros((indice,columna_variable+2))

for i in range(indice):
	for j in range(columna_variable):
		matriz_final[i][j] = matriz_reglas[i][j]


for i in range(len(matriz_final)):
	p = pesos_candidatos(train,matriz_final[i],y_train)
	p = peso_regla(p)
	matriz_final[i][columna_variable] = p[0]
	matriz_final[i][columna_variable+1] = p[1]

print(matriz_final)
print(len(matriz_final))

# Probamos la funcion test
etiquetas_train = []
etiquetas_test = []

for i in range(len(train)):
	resultado = test(train[i],matriz_final)
	
	etiquetas_train.append(int(resultado))

print(confusion_matrix(y_train, etiquetas_train))
print(accuracy_score(y_train, etiquetas_train))



for i in range(len(dataset_test)):
	resultado = test(dataset_test[i],matriz_final)
	etiquetas_test.append(int(resultado))


print(confusion_matrix(y_test,etiquetas_test))
print(accuracy_score(y_test, etiquetas_test))
print(f1_score(y_test, etiquetas_test, average='macro'))
#print("la clase correspondiente es la clase ", label[int(resultado[1])], " y tiene un peso asociado igual a ", resultado[0])

