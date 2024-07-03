import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import copy
import math
import random
from scipy.spatial.distance import hamming as ham
#Variables fijas
columna_variable = 7
chi = 3
chi_excep = 5
## Leer los documentos
file = '1tra.csv'
data = pd.read_csv(file,sep=',',header=None).values
train = data[:, 0:-1]
y_train = data[:,-1].astype(int)

file2 = '1tst.csv'
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
	chi4 = 3
	columna = int(columna)
	c = int(c)
	yy = columna * chi4 + c
	yy = int(yy)

	if(m_p[yy][0] <= valor and valor <= m_p[yy][1]):
		
		if (m_p[yy][0] == valor and m_p[yy][1] == valor):
		
			return 1.0
			
		else:
			
			return (valor - m_p[yy][0])/(m_p[yy][1] - m_p[yy][0])
			
	elif(m_p[yy][1] <= valor and valor <= m_p[yy][2]):
		
		if(m_p[yy][2] == valor and m_p[yy][1] == valor):
			
			return 1.0
			
		else:
			
			return (m_p[yy][2]- valor)/(m_p[yy][2] - m_p[yy][1])
			
	else:
			
		return 0
			

def valor_pertenencia3(valor3,columna3,c3,m_p3):
	
		i3 = columna3 * chi_excep + c3

		if(m_p3[i3][0] <= valor3 and valor3 <= m_p3[i3][1]):
		
			if (m_p3[i3][0] == valor3 and m_p3[i3][1] == valor3):
		
				return 1.0
			
			else:
			
				return (valor3 - m_p3[i3][0])/(m_p3[i3][1] - m_p3[i3][0])
			
		elif(m_p3[i3][1] <= valor3 and valor3 <= m_p3[i3][2]):
		
			if(m_p3[i3][2] == valor3 and m_p3[i3][1] == valor3):
			
				return 1.0
			
			else:
			
				return (m_p3[i3][2]- valor3)/(m_p3[i3][2] - m_p3[i3][1])
			
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

#	print(suma)
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


def detectar(matriz,corte):
	resultado = []
	for i in range(len(matriz)):
		if(matriz[i][columna_variable] < corte):
			resultado.append(matriz[i])
	return resultado

def eliminar(matriz,corte):

	matriz2 = []
	for i in range(len(matriz)):

		if(matriz[i][columna_variable] > corte):
			matriz2.append(matriz[i])

			

	return matriz2

def transformar_en_matriz(lista):
	matriz = np.zeros((len(lista),len(lista[0])))

	for i in range(len(lista)):
		for j in range(len(lista[0])):
			matriz[i][j] = lista[i][j]

	return matriz

def crear_train(vectoran):
	new_train = []
	lista_etiquetas = []
	for i in range(len(train)):
		contador = 0 
		for j in range(columna_variable):

			por = valor_pertenencia(train[i][j],j,vectoran[j])
			if(por >= 0.5):
				contador = contador + 1
		if(contador == columna_variable):
			new_train.append(train[i])
			lista_etiquetas.append(y_train[i])

	resultado = []
	resultado.append(new_train)
	resultado.append(lista_etiquetas)
	return resultado



########################################################## Inicio  #########################
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
#print("Matriz de pertenencia")
#print(m_p)
matrix_pertenencia = np.zeros((len(train), columna_variable))

# De aqui para abajo arreglar ahora mismo generamos bien las reglas pero no generamos bien los pesos

matriz_reglas = np.zeros((len(train), columna_variable+2))

for i in range(len(train)):
	matriz_reglas[i][0] = 7

for i in range(len(train)):
	for j in range(columna_variable):

		vector_asociado = valor_maximo(train[i][j],j)
		matrix_pertenencia[i][j]= vector_asociado[0]


#print(" ")
#print("Matriz de ejemplos")
#print(matrix_pertenencia)
#print(len(matrix_pertenencia))


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
#etiquetas_train = []
#etiquetas_test = []

#for i in range(len(train)):
#	resultado = test(train[i],matriz_final)
	
#	etiquetas_train.append(int(resultado))

#print(confusion_matrix(y_train, etiquetas_train))
#print(accuracy_score(y_train, etiquetas_train))



#for i in range(len(dataset_test)):
#	resultado = test(dataset_test[i],matriz_final)
#	etiquetas_test.append(int(resultado))


#print(confusion_matrix(y_test,etiquetas_test))
#print(accuracy_score(y_test, etiquetas_test))
#print("la clase correspondiente es la clase ", label[int(resultado[1])], " y tiene un peso asociado igual a ", resultado[0])

#print(detectar(matriz_final,0.8))
#print(len(detectar(matriz_final,0.8)))

matriz_detec = detectar(matriz_final,0.8)
matriz_detec = transformar_en_matriz(matriz_detec)
#matriz_res = eliminar(matriz_final,0.8) 
#matriz_res = transformar_en_matriz(matriz_res)

#print(matriz_detec)
#print(len(matriz_detec))


excepcio = []
excepcio_label = []
for tt in range(len(matriz_detec)):

	entrenamiento_nuevo = crear_train(matriz_detec[tt])
	etiquetas_nuevas = entrenamiento_nuevo[1]
	entrenamiento_nuevo = transformar_en_matriz(entrenamiento_nuevo[0])	
	excepcio.append(entrenamiento_nuevo)
	excepcio_label.append(etiquetas_nuevas)


#print(etiquetas_nuevas)
#print(entrenamiento_nuevo)

##################################################################### Fin del CHI 3 y separacion de conjuntos ##############

class CHI:

	def __init__(self,vectores,conjuntotrain,etique,chi1):
		self.v = vectores
		self.ct = conjuntotrain
		self.e = etique
		self.chi2 = chi1

	def get_ct(self):
		return self.ct
	def get_chi2(self):
		return self.chi2
	def get_v(self):
		return self.v
	def get_e(self):
		return self.e

	def valor_pertenencia2(self,valor2,columna2,c2,m_p2):
	
		i2 = columna2 * self.get_chi2() + c2

		if(m_p2[i2][0] <= valor2 and valor2 <= m_p2[i2][1]):
		
			if (m_p2[i2][0] == valor2 and m_p2[i2][1] == valor2):
		
				return 1.0
			
			else:
			
				return (valor2 - m_p2[i2][0])/(m_p2[i2][1] - m_p2[i2][0])
			
		elif(m_p2[i2][1] <= valor2 and valor2 <= m_p2[i2][2]):
		
			if(m_p2[i2][2] == valor2 and m_p2[i2][1] == valor2):
			
				return 1.0
			
			else:
			
				return (m_p2[i2][2]- valor2)/(m_p2[i2][2] - m_p2[i2][1])
			
		else:
			
			return 0

	def matriz_pertenencia2(self,m_maxmin2):
		filas2 = (self.get_chi2()*(columna_variable))
		index2 = 0
		ind2 = 0
		m_p2 = np.zeros((filas2, 5))

		for i in range(columna_variable):
			for j in range(self.get_chi2()):
				base2 = (m_maxmin2[i][1] - m_maxmin2[i][0]) / (self.get_chi2() - 1)
				filamp2 = self.get_chi2()*i+j 
				if((j%self.get_chi2()) == 0 ):
					m_p2[filamp2][0] =  m_maxmin2[i][0]
					m_p2[filamp2][1] =  m_maxmin2[i][0]
					m_p2[filamp2][2] =  m_maxmin2[i][0] + base2
				elif((j%self.get_chi2()) == (self.get_chi2() -1)):
					m_p2[filamp2][0] = m_p2[filamp2-1][1]
					m_p2[filamp2][1] =  m_maxmin2[i][1]
					m_p2[filamp2][2] =  m_maxmin2[i][1]
				else:
					m_p2[filamp2][0] = m_p2[filamp2-1][1]
					m_p2[filamp2][1] = m_p2[filamp2-1][2]
					m_p2[filamp2][2] = m_p2[filamp2-1][2] + base2 

		
				m_p2[filamp2][3] = i
				m_p2[filamp2][4] = j

		return m_p2

	def valor_maximo2(self,valor2,columna2,m_p2):

		resultado2 = []
		valor_max2 = -50
		clase2 = -50
		for i in range(self.get_chi2()):
			v_resultados2 = self.valor_pertenencia2(valor2,columna2,i,m_p2)
			if(v_resultados2 > valor_max2):
				valor_max2 = v_resultados2
				clase2 = i

	
		resultado2.append(clase2)
		resultado2.append(valor_max2)
	
		return resultado2

	def vec_in_vec2(self,vector2,v_matriz2): 

		contador2 = 0
		for i in range(columna_variable):
			if(vector2[i] == v_matriz2[i]):
				contador2 = contador2 + 1
				if(contador2 == columna_variable):
					return True
		return False

	def pesos_candidatos2(self,matriz_train2,vec_re2,etiquetas2,m_p2):

		suma2 = []
		for k in range(len(label)):
			suma2.append(0)
	

		for i in range(len(matriz_train2)):
			rmax2 = 1

			for j in range(columna_variable):
			
				r2 = self.valor_pertenencia2(matriz_train2[i][j],j,int(vec_re2[j]),m_p2)
			
			
				rmax2 = rmax2 * r2


			suma2[int(etiquetas2[i])] = suma2[int(etiquetas2[i])] + rmax2 

		#print(suma2)
		return suma2

	def peso_regla2(self,pesos_candidatos2):

		maximo2 = -1
		total2 = 0
		clase2 = -1
		for i in range(len(pesos_candidatos2)):
			total2 = pesos_candidatos2[i] + total2
		for i in range(len(pesos_candidatos2)):
			if(i == self.v[columna_variable+1]):
				total2 = total2
			elif((pesos_candidatos2[i]/total2) > maximo2):
				maximo2 = pesos_candidatos2[i]/total2
				clase2 = i
		resultado2 = []
		resultado2.append(maximo2)
		resultado2.append(clase2)
		return resultado2

	def inicio(self):
		ct2 = self.get_ct()
		max_min2 = np.zeros((columna_variable, 3))
		for i in range(columna_variable):
			max_min2[i][0] = 100000


		for j in range(columna_variable):
			

			for i in range(len(ct2)):
				if(max_min2[j][0] > ct2[i][j]):
					max_min2[j][0] = ct2[i][j]
				if(max_min2[j][1] < ct2[i][j]):
					max_min2[j][1] = ct2[i][j]
				
			max_min2[j][2]=j

		m_p2 = self.matriz_pertenencia2(max_min2)
		#print(m_p2)
		matrix_pertenencia2 = np.zeros((len(ct2), columna_variable))
		matriz_reglas2 = np.zeros((len(ct2), columna_variable+2))

		for i in range(len(ct2)):
			matriz_reglas2[i][0] = 7

		for i in range(len(ct2)):
			for j in range(columna_variable):

				vector_asociado2 = self.valor_maximo2(ct2[i][j],j,m_p2)
				matrix_pertenencia2[i][j]= vector_asociado2[0]

		#print(" ")
		#print("Matriz de ejemplos")
		#print(matrix_pertenencia2)
		#print(len(matrix_pertenencia2))

		indice2 = 0

		for i in range(len(ct2)):
			contador2 = 0
			for j in range(len(ct2)):
				if(self.vec_in_vec2(matriz_reglas2[j],matrix_pertenencia2[i]) == False):
					contador2 = contador2 + 1
					if(contador2 == len(ct2)):
						for k in range(columna_variable):
							matriz_reglas2[indice2][k] = (matrix_pertenencia2[i][k])
						indice2 = indice2 + 1

		matriz_final2 = np.zeros((indice2,columna_variable+2))

		for i in range(indice2):
			for j in range(columna_variable):
				matriz_final2[i][j] = matriz_reglas2[i][j]


		for i in range(len(matriz_final2)):
			p2 = self.pesos_candidatos2(ct2,matriz_final2[i],self.get_e(),m_p2)
			p2 = self.peso_regla2(p2)
			matriz_final2[i][columna_variable] = p2[0]
			matriz_final2[i][columna_variable+1] = p2[1]

		#print(m_p2)
		#print(matriz_final2)
		matrizr2 = []
		nun_re = 0
		listado = []
		for i in range(len(matriz_final2)):
			ve = self.get_v()
			if(matriz_final2[i][columna_variable+1] == ve[columna_variable+1] or matriz_final2[i][columna_variable] < 0.9): #cambiar por el peso de la regla original
				nun_re = nun_re
			else:
				nun_re = nun_re + 1
				listado.append(i)

		matriz_repu = np.zeros((nun_re, columna_variable+2))			
		for i in range(len(listado)):
			for j in range(columna_variable+2):
				matriz_repu[i][j]=matriz_final2[listado[i]][j]

		#print("funciones de pertenencia")
		#print(m_p2)
		#print("matriz de reglas")
		#print(matriz_repu)
		repu = []
		repu.append(nun_re)
		repu.append(m_p2)    ###Anadir funciones de pertenencia
		repu.append(matriz_repu)   ###Añadir matriz final
		return repu


excepcion = []
for tt in range(len(matriz_detec)):
	x = CHI(matriz_detec[tt],excepcio[tt],excepcio_label[tt],chi_excep)
	excepcion.append(x)

Numero_reglas = len(matriz_final)
lista_reglas =[]
lista_reglas.append(matriz_final)
lista_funciones = []
lista_funciones.append(m_p)
n_reglas = []
n_reglas.append(Numero_reglas)
n_reglas_acum = []
n_reglas_acum.append(Numero_reglas)

for i in range(len(excepcion)):
	n = excepcion[i].inicio()
	Numero_reglas = Numero_reglas + n[0]
	n_reglas_acum.append(Numero_reglas)
	n_reglas.append(n[0])
	lista_funciones.append(n[1])
	lista_reglas.append(n[2])

#print("el numero de reglas es igual a", Numero_reglas)


def test_excep(vector_ejemplo,f_pertenencia,rules):
	claase = 0
	valoor = 0
		
	for i in range(len(rules)):

		res = 50
		for j in range(columna_variable):
			candidato = valor_pertenencia3(vector_ejemplo[j],j,int(rules[i][j]),f_pertenencia)
			if(candidato <= res):
				res = candidato

		if(candidato >= 0.7  and rules[i][columna_variable] >= 0.9):
			res = res * rules[i][columna_variable]
			if(valoor < res):
				valoor = res
				claase = rules[i][columna_variable+1]	
	l = []
	l.append(valoor) 
	l.append(claase) 
	return l


def test(vec,ma):

	cl = -50
	val = -50

	for i in range(len(excepcion)):
		if(i > 0):
			prueba = test_excep(vec,lista_funciones[i],lista_reglas[i])
			if(val < prueba[0]):
				val = prueba[0]
				cl = prueba[1]

	if(val > 0):
		return cl
	
	for i in range(len(ma)):

		res = 50
		for j in range(columna_variable):
			candidato = valor_pertenencia(vec[j],int(j),int(ma[i][j]))
			if(candidato <= res):
				res = candidato

		res = res * ma[i][columna_variable]
		
		if(val < res ):
			val = res
			cl = ma[i][columna_variable+1]


	return cl

###################################################################FIN CALCULO DE EXCEPCIONES##################################
###################################################################INICIO CHC##################################################
print("el numero de reglas es igual a", Numero_reglas)
print(n_reglas)
print(n_reglas_acum)
#print(lista_reglas)
#print(lista_funciones)

class CHC:

    def __init__(self):  ##cambiarlo por el fichero train y las reglas
    	self.tam_reg = int(Numero_reglas)
    	self.tam_2t = int(chi*columna_variable + chi_excep * columna_variable * (len(lista_reglas)-1))
    	self.tam_crom = self.tam_reg + self.tam_2t
    	print ("Numero instancias: " + str(self.tam_crom))
    	self.crom_ini = []
    	for i in range(self.tam_reg):
    		self.crom_ini.append(1)
    	for e in range(self.tam_2t):
    		self.crom_ini.append(0)
    	print ("Cromosoma inicial ")
    	print (self.crom_ini)

    def init_P(self,t):
    	print("Poblacion Inicial ")
    	self.pob = {}
    	crom = []
    	self.tam = t

    	for i in range(self.tam):
    		for e in range(self.tam_reg):
    			crom.append(random.randrange(0,2))
    		
    		for e in range(self.tam_2t):
    			crom.append(random.uniform(-0.5,0.5))
    		self.pob[i] = crom
    		crom = []

    	#print (self.pob)
    	return self.pob, self.tam_crom

    def hux(self, p, u = 0): #genero la descendencia
        if u == 0:
            u = self.tam_crom/4 #umbral de apareamiento

        print ("Umbrar de apareamiento: " + str(u))


        self.p_d = {}
        aux = 0
        for n in range(int(self.tam/2)): #es el número de la poblacion por tanto se aparean por parejas de ahi la mitad
        	p1 = p[random.randrange(0, len(p))] #se toman los individuos
        	p2 = p[random.randrange(0, len(p))]

        	if ( (ham(p1, p2) * self.tam_crom)  > u): #vemos si se aparean
        		m = (ham(p1, p2) * self.tam_crom) / 2 #cuantas veces se mezclan
        		m = int(m)
        		h1 = p1
        		h2 = p2

        		while (m > 0):
        			bit_p = random.randrange(0, len(p1))
        			if (p1[bit_p] != p2[bit_p]) and (p1[bit_p] != h2[bit_p]):
        				# Crear decendiente de p1, p2
        				aux1 = p2[bit_p]
        				aux2 = p1[bit_p]
        				h1[bit_p] = aux1
        				h2[bit_p] = aux2
        				m -= 1
        		self.p_d[aux] = h1
        		self.p_d[aux + 1] = h2
        		aux += 1

        print("Descendencia")
        print (self.p_d)
        return self.p_d


    def valor_pertenencia_2tt(self,valor,columna,c,tt,m_p2tt,chii):
    	i = columna * chii + c
    	izq = m_p2tt[i][0] + (m_p2tt[i][1]-m_p2tt[i][0])*tt
    	cen = m_p2tt[i][1] + (m_p2tt[i][1]-m_p2tt[i][0])*tt
    	der = m_p2tt[i][2] + (m_p2tt[i][1]-m_p2tt[i][0])*tt
    	if(izq <= valor and valor <= cen):

    		if (izq == valor and cen == valor):

    			return 1.0

    		else:

    			return (valor - izq)/(cen - izq)

    	elif(cen <= valor and valor <= der):

    		if(der == valor and cen == valor):

    			return 1.0

    		else:

    			return (der- valor)/(der - cen)

    	else:

    		return 0


    def test_excep7(self,vector_ejemplo,f_pertenencia,rules,cromm,position): #hay que adaptarlas a nuestra situacion
    	claase = 0
    	valoor = 0
    	for i in range(len(rules)):
    		poss = n_reglas_acum[position-1] + i
    		if(cromm[poss] == 1):
    			res = 50
    			for j in range(columna_variable):
    				poss_t = Numero_reglas + chi * columna_variable + (position-1)*chi_excep*columna_variable+j*chi_excep + int(rules[i][j])
    				candidato = self.valor_pertenencia_2tt(vector_ejemplo[j],j,int(rules[i][j]),cromm[poss_t],f_pertenencia,chi_excep)
    				if(candidato <= res):
    					res = candidato

    			if(candidato >= 0.7  and rules[i][columna_variable] >= 0.9):
    				res = res * rules[i][columna_variable]
    				if(valoor < res):
    					valoor = res
    					claase = rules[i][columna_variable+1]	
    	
    	l = []
    	l.append(valoor)
    	l.append(claase)
    	return l

    def test7(self,vector,ma,croo): #hay que adaptarlas a nuestra situacion

    	cl = -50
    	val = -50

    	for i in range(len(lista_reglas)):
    		if(i > 0):
    			prueba = self.test_excep7(vector,lista_funciones[i],lista_reglas[i],croo,i)
    			if(val < prueba[0]):
    				val = prueba[0]
    				cl = prueba[1]

    	if(val > 0):
    		return cl

    	for i in range(len(ma)):
    		if(croo[i] == 1):
    			res = 50
    			for j in range(columna_variable):
    				candidato = self.valor_pertenencia_2tt(vector[j],j,int(ma[i][j]),croo[int(Numero_reglas+ (j * chi + int(ma[i][j])))],lista_funciones[0],chi)
    				if(candidato <= res):
    					res = candidato

    			res = res * ma[i][columna_variable]

    			if(val < res ):
    				val = res
    				cl = ma[i][columna_variable+1]

    	return cl

    def eval_cromo(self,cromosos):
    	etiquetas_train = []
    	for i in range(len(train)):
    		resultado = self.test7(train[i],matriz_final,cromosos)
    		etiquetas_train.append(int(resultado))

    	resul = f1_score(y_train, etiquetas_train,average='macro')
    	return resul

    def eval_P(self, p):
    	sultados =[]
    	for i in range(len(p)):
    		sultados.append(self.eval_cromo(p[i]))
    	return sultados

	#Seleccion elitista
    def sel_eti(self,p,h):
        self.p_n = {}

        print("Seleccion elitista")
       
        punt_p = self.eval_P(p)
        punt_h = self.eval_P(h)

        pob_total = {}
        l_aux = []
        cont = 0
        for key in p:
            l_aux.append(p[key])
            l_aux.append(punt_p[key])
            pob_total[cont] = l_aux
            l_aux = []
            cont += 1

        for key in h:
            l_aux.append(h[key])
            l_aux.append(punt_h[key])
            pob_total[cont] = l_aux
            l_aux = []
            cont += 1

        print("Poblacion total")
        print(pob_total)


        max = 0
        key_d = None

        for e in range(self.tam):
            for key in pob_total:
                if (pob_total[key][1] >= max):
                    max = pob_total[key][1]
                    key_d = key
            self.p_n[e] = pob_total[key_d]
            max = 0
            del (pob_total[key_d])

        print("Nueva poblacion")
        p_nueva = {}

        for key in self.p_n:
            p_nueva[key] = self.p_n[key][0]

        print (self.p_n)

        #-----------

        return p_nueva, self.p_n


    #Reinicializacion
    def reinicializacion(self, p, umb):

        p_r = {}
        max = 0
        crom_max = []
        print("Reinicializacion")
        for key in p:
            if p[key][1] > max:
                max = p[key][1]
                crom_max = p[key][0]

        print ("Mejor Individuo(cromosoma)")
        print (str(crom_max) + " , " + str(max))

        porc =  int(100 * float (umb) /float (self.tam_crom))
        print ("::" + str(porc))

        crom = copy.deepcopy(crom_max)


        for i in range(self.tam):
            crom = copy.deepcopy(crom_max)
            for j in range(porc):
                crom[random.randrange(0, self.tam_crom)] = random.randrange(0,2)


            p_r[i] = crom



        print ("Nueva poblacion")
        print(p_r)
        return p_r

etiquetas_train = []
#etiquetas_test = []



for i in range(len(train)):
	resultado = test(train[i],matriz_final)
	
	etiquetas_train.append(int(resultado))

#print(confusion_matrix(y_train, etiquetas_train))
#cort = accuracy_score(y_train, etiquetas_train)



#for i in range(len(dataset_test)):
#	resultado = test(dataset_test[i],matriz_final)
#	etiquetas_test.append(int(resultado))


#print(confusion_matrix(y_test,etiquetas_test))
#print(accuracy_score(y_test, etiquetas_test))
#print(f1_score(y_test, etiquetas_test, average='macro'))
umb_r = 3.5


g = CHC()
p, tam_c = g.init_P(61)

d = tam_c / 4

for i in range(100):
    p_aux = copy.deepcopy(p)
    desc = g.hux(p, d)
    p, p_eva = g.sel_eti(p_aux,desc)

    if (len(desc)) == 0:
        d = d - 1

    if (d == 0) :
        print("*** Reinicializacion ***")
        print("Iteracion : " + str(i))
        p = g.reinicializacion(p_eva, umb_r )
        d = tam_c / 4

print("p")
print(" ")
print(p)


def valor_pertenencia_2tt1(valor,columna,c,tt,m_p2tt,chii):
  	i = columna * chii + c
  	izq = m_p2tt[i][0] + (m_p2tt[i][1]-m_p2tt[i][0])*tt
  	cen = m_p2tt[i][1] + (m_p2tt[i][1]-m_p2tt[i][0])*tt
  	der = m_p2tt[i][2] + (m_p2tt[i][1]-m_p2tt[i][0])*tt
  	if(izq <= valor and valor <= cen):
  		if (izq == valor and cen == valor):

  			return 1.0

  		else:

  			return (valor - izq)/(cen - izq)

  	elif(cen <= valor and valor <= der):

  		if(der == valor and cen == valor):

  			return 1.0

  		else:

  			return (der- valor)/(der - cen)

  	else:

  		return 0

def test_excep71(vector_ejemplo,f_pertenencia,rules,cromm,position): #hay que adaptarlas a nuestra situacion
	claase = 0
	valoor = 0
	for i in range(len(rules)):
		poss = n_reglas_acum[position-1] + i
		if(cromm[poss] == 1):
			res = 50
			for j in range(columna_variable):
				poss_t = Numero_reglas + chi * columna_variable + (position-1)*chi_excep*columna_variable+j*chi_excep + int(rules[i][j])
				candidato = valor_pertenencia_2tt1(vector_ejemplo[j],j,int(rules[i][j]),cromm[poss_t],f_pertenencia,chi_excep)
				if(candidato <= res):
					res = candidato

			if(candidato >= 0.7  and rules[i][columna_variable] >= 0.9):
				res = res * rules[i][columna_variable]
				if(valoor < res):
					valoor = res
					claase = rules[i][columna_variable+1]

	l = []
	l.append(valoor)
	l.append(claase)
	return l

def test71(vector,ma,croo): #hay que adaptarlas a nuestra situacion
	cl = -50
	val = -50
	print(croo)
	for i in range(len(lista_reglas)):
		if(i > 0):
			prueba = test_excep71(vector,lista_funciones[i],lista_reglas[i],croo,i)
			if(val < prueba[0]):
				val = prueba[0]
				cl = prueba[1]

	if(val > 0):
		return cl

	for i in range(len(ma)):
		if(croo[i] == 1):
			res = 50
			for j in range(columna_variable):
				candidato = valor_pertenencia_2tt1(vector[j],j,int(ma[i][j]),croo[int(Numero_reglas+ (j * chi + int(ma[i][j])))],lista_funciones[0],chi)
				if(candidato <= res):
					res = candidato

			res = res * ma[i][columna_variable]

			if(val < res ):
				val = res
				cl = ma[i][columna_variable+1]

	return cl

def cuenta_reglas(vector):
	con = 0
	for i in range(len(vector)):
		if( vector[i] == 1):
			con = con + 1
	return con

regla_max = Numero_reglas
vct = []
cort=0
rltado = g.eval_P(p)
for i in range(len(p)):
	if((rltado[i] > cort)):
		vct = p[i]
		regla_max = cuenta_reglas(p[i])
		cort = rltado[i]
	elif((rltado[i] == cort) and (regla_max > cuenta_reglas(p[i]))):
		vct = p[i]
		regla_max = cuenta_reglas(p[i])
		cort = rltado[i]

co = 0
for i in range(len(matriz_final)):
	if(vct[i] == 1):
		co = co + 1	

etiquetas_test = []

for i in range(len(dataset_test)):
	resultado = test71(dataset_test[i],matriz_final,vct)
	etiquetas_test.append(int(resultado))


print("***************")
print("Terminacion")
print("***************")
print("el vector elegido es: ")
print(vct)
print("El train tiene un macro F1 igual a: ")
print(cort)
print("El numero de reglas base es igual a: ")
print(co)
coex = regla_max - co
print("El numero de reglas excepcion es igual a: ")
print(coex)
print("El numero de reglas es igual a: ")
print(regla_max)
print("La matriz de confusion es igual a: ")
print(confusion_matrix(y_test,etiquetas_test))
print("El accuracy test es igual a: ")
print(accuracy_score(y_test, etiquetas_test))
print("El test tiene un macro F1 igual a:  ")
print(f1_score(y_test, etiquetas_test, average='macro'))
print("***************")
print("Terminacion")
print("***************")