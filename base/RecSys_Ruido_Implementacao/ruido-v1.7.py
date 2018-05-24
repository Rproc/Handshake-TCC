# coding: utf-8
import collections as ct
import numpy as np
import random
import math
import time
from copy import copy
from operator import itemgetter
import knn
import svd
import matplotlib.pyplot as plt

COSSENO = 'cos'
EUCLIDIANA = 'euc'
MANHATTAN = 'man'

def contar(dataset,n_usuarios, n_filmes):
	qtd = 0
	for i in range(n_usuarios):
		for j in range(n_filmes):
			if dataset[i,j] > 0.1:
				qtd += 1

	print(qtd)

def contar2(dataset1,dataset2,n_usuarios, n_filmes):
	qtd = 0
	dif = 0
	for i in range(n_usuarios):
		for j in range(n_filmes):
			if dataset1[i,j] > 0.1:
				qtd += 1
				if dataset1[i,j] != dataset2[i,j]:
					dif += 1

	porc = dif/qtd
	print(porc,"qtd bruta: ",dif)


def criar_datasets(n_usuarios, n_filmes, porcentagem_treino, n_ratings):
	dataset = np.zeros((n_usuarios,n_filmes), dtype=np.float)
	d_treino = np.zeros((n_usuarios,n_filmes), dtype=np.float)
	d_teste = np.zeros((n_usuarios, n_filmes), dtype=np.float)
	indices_teste = []
	indices_treino = []
	timestamp = np.zeros(((int(n_ratings*(porcentagem_treino/100)) ),3), dtype=np.int)

	i_arranjo = 0
	i = 0

	texto = open("ml-100k/u.data","r")
	linhas = texto.readlines()
	n_ratings =	len(linhas)
	limite_treino = n_ratings * (porcentagem_treino/100)

	# cria vetor de permutacao aleatoria de 0 a n_ratings-1
	arranjo = np.random.permutation(n_ratings)

	for linha in linhas:
		info = linha.split("\t")
		i_usuario = int(info[0])-1
		i_filme = int(info[1])-1

		dataset[i_usuario, i_filme] = int(info[2])

		if arranjo[i_arranjo] < limite_treino:
			d_treino[i_usuario, i_filme] = info[2]
			indices_treino.append((i_usuario, i_filme))
			if(d_treino[i_usuario, i_filme] > 0):
				timestamp[i, 0] = int(info[3])
				timestamp[i, 1] = i_usuario
				timestamp[i, 2] = i_filme
				i += 1
		else:
			d_teste[i_usuario, i_filme] = info[2]
			# print(d_teste[i_usuario, i_filme])
			if(int(info[2]) > 0):
				indices_teste.append((i_usuario, i_filme))

		i_arranjo += 1

	# print("Criar ", contar(d_teste, n_usuarios, n_filmes))
	contar(dataset,n_usuarios,n_filmes)
	return dataset, d_treino, d_teste, indices_treino, indices_teste, timestamp


def construir_treino_por_timestamp(d_treino, n_batches, tamanho_batch, n_usuarios, n_filmes, timestamp, n_ratings, porcentagem_treino):
	d_treino_t = np.zeros((n_usuarios,n_filmes), dtype=np.float)
	indices_treino_t = []
	i_rating_usuarios = np.zeros(n_usuarios, dtype=np.int)
	qtd_rating = 0
	u = 0
	i = 0

	timestamp_ord = sorted(timestamp, key=lambda x: x[0])

	# np.reshape(timestamp_ord, (  int(n_ratings*(porcentagem_treino/100)), 3  )   )
	ts_ord = np.asarray(timestamp_ord)
	# print(type(ts_ord))

	indices_timestamp_atual = 0

	while qtd_rating < tamanho_batch:
		# print (ts_ord[0,1])
		# input()
		u = ts_ord[indices_timestamp_atual, 1]
		i = ts_ord[indices_timestamp_atual, 2]
		d_treino_t[u, i] = d_treino[u, i]
		indices_treino_t.append((u, i))

		indices_timestamp_atual += 1
		qtd_rating += 1

	return d_treino_t, indices_treino_t, indices_timestamp_atual, ts_ord


def atualizar_treino_timestamp(d_treino, n_batches, tamanho_batch, d_treino_t, indices_treino_t, tamanho_treino, t, timestamp_ord, indices_timestamp_atual):

	qtd_rating = 0

	if t == n_batches - 1:
		tamanho_batch += tamanho_treino % n_batches

	while qtd_rating < tamanho_batch:
		usuario = timestamp_ord[indices_timestamp_atual, 1]
		item = timestamp_ord[indices_timestamp_atual, 2]
		d_treino_t[usuario, item] = d_treino[usuario, item]
		indices_treino_t.append((usuario, item))

		indices_timestamp_atual += 1
		qtd_rating += 1

	return d_treino_t, indices_treino_t, indices_timestamp_atual

def construir_treino_por_tempo(d_treino, n_batches, tamanho_batch, n_usuarios, n_filmes):
	d_treino_t = np.zeros((n_usuarios,n_filmes), dtype=np.float)
	indices_treino_t = []
	i_rating_usuarios = np.zeros(n_usuarios, dtype=np.int)
	qtd_rating = 0
	usuario_atual = 0
	encontrou_rating = False

	while qtd_rating < tamanho_batch:
		if i_rating_usuarios[usuario_atual] < n_filmes:
			if d_treino[usuario_atual, i_rating_usuarios[usuario_atual]] > 0.1:
				d_treino_t[usuario_atual, i_rating_usuarios[usuario_atual]] = d_treino[usuario_atual, i_rating_usuarios[usuario_atual]]
				indices_treino_t.append((usuario_atual, i_rating_usuarios[usuario_atual]))
				i_rating_usuarios[usuario_atual] += 1
				qtd_rating += 1
			else:
				i_rating_usuarios[usuario_atual] += 1

				while not encontrou_rating:

					if i_rating_usuarios[usuario_atual] < n_filmes:
						if d_treino[usuario_atual, i_rating_usuarios[usuario_atual]] > 0.1:
							d_treino_t[usuario_atual, i_rating_usuarios[usuario_atual]] = d_treino[usuario_atual, i_rating_usuarios[usuario_atual]]
							indices_treino_t.append((usuario_atual, i_rating_usuarios[usuario_atual]))
							encontrou_rating = True
							qtd_rating += 1
					else:
						encontrou_rating = True
					i_rating_usuarios[usuario_atual] += 1

		usuario_atual = (usuario_atual + 1) % n_usuarios

	return d_treino_t, indices_treino_t, i_rating_usuarios, usuario_atual


def flip_noise(d_treino_t, porcentagem_ruido, tamanho_batch, t, indices_treino_t):
	treino_ruidoso = copy(d_treino_t)
	qtd_flips = int((len(indices_treino_t) ) * (porcentagem_ruido/100))
	arranjo = np.random.permutation(len(indices_treino_t))

	for i in range(qtd_flips):
		usuario = indices_treino_t[arranjo[i]][0]
		item = indices_treino_t[arranjo[i]][1]
		treino_ruidoso[usuario,item] = flip(d_treino_t[usuario,item])

	return treino_ruidoso


def flip(nota):
	if nota <= 3:
		return random.randint(4, 5)
	else:
		return random.randint(1, 3)


def atualizar_treino_t(d_treino, n_batches, tamanho_batch, n_usuarios, n_filmes, d_treino_t, indices_treino_t, i_rating_usuarios, usuario_atual, tamanho_treino, t):

	qtd_rating = 0
	encontrou_rating = False

	if t == n_batches - 1:
		tamanho_batch += tamanho_treino % n_batches

	while qtd_rating < tamanho_batch:
		if i_rating_usuarios[usuario_atual] < n_filmes:
			if d_treino[usuario_atual, i_rating_usuarios[usuario_atual]] > 0.1:
				d_treino_t[usuario_atual, i_rating_usuarios[usuario_atual]] = d_treino[usuario_atual, i_rating_usuarios[usuario_atual]]
				indices_treino_t.append((usuario_atual, i_rating_usuarios[usuario_atual]))
				i_rating_usuarios[usuario_atual] += 1
				qtd_rating += 1
			else:
				i_rating_usuarios[usuario_atual] += 1

				while not encontrou_rating:

					if i_rating_usuarios[usuario_atual] < n_filmes:
						if d_treino[usuario_atual, i_rating_usuarios[usuario_atual]] > 0.1:
							d_treino_t[usuario_atual, i_rating_usuarios[usuario_atual]] = d_treino[usuario_atual, i_rating_usuarios[usuario_atual]]
							indices_treino_t.append((usuario_atual, i_rating_usuarios[usuario_atual]))
							encontrou_rating = True
							qtd_rating += 1
					else:
						encontrou_rating = True
					i_rating_usuarios[usuario_atual] += 1

		usuario_atual = (usuario_atual + 1) % n_usuarios

	return d_treino_t, indices_treino_t, i_rating_usuarios, usuario_atual


def main():

	n_usuarios = 943
	n_filmes = 1682
	porcentagem_treino = 90
	# porcentagem de ruido inicial, que sera incrementado com o tempo
	porcentagem_ruido = 5
	n_batches = 10
	n_ratings = 100000
	incremento_ruido = 5
	n_ruidos = 6
	iteracao = 0

	# matriz que vai receber os rmse pra plotar os graficos no final
	# v_rmseK = np.zeros((n_ruidos+1, n_batches), dtype=float)
	v_rmseS = np.zeros((n_ruidos+1, n_batches), dtype=float)

	tamanho_treino = n_ratings * (porcentagem_treino/100)
	tamanho_batch = tamanho_treino//n_batches

	# contar(d_treino_t, n_usuarios, n_filmes) >testando a divisao dos datasets
	tipo_similaridade = COSSENO
	dataset, d_treino, d_teste, indices_treino, indices_teste, timestamp = criar_datasets(n_usuarios, n_filmes, porcentagem_treino, n_ratings)
	print("tamanho_batch: ", tamanho_batch, " tamanho_treino: " , tamanho_treino," n_batches: ", n_batches)
	# for iteracao in range(4):


	t = 0
	# constroi o primeiro batch (momento inicial)
	# d_treino_t, indices_treino_t, i_rating_usuarios, usuario_atual = construir_treino_por_tempo(d_treino, n_batches, tamanho_batch, n_usuarios, n_filmes)
	d_treino_t, indices_treino_t, indices_timestamp_atual, timestamp_ord = construir_treino_por_timestamp(d_treino, n_batches, tamanho_batch, n_usuarios, n_filmes, timestamp, n_ratings, porcentagem_treino)
	while t < n_batches:
		print("t ",t,"\n  sem ruido...")


		# maeKnn, rmseKnn = knn.knn(d_treino_t, d_teste, n_usuarios, n_filmes, indices_teste, tipo_similaridade)
		maeSVD, rmseSVD = svd.svd(d_treino_t, d_teste, n_usuarios, n_filmes, indices_teste, indices_treino_t)
		# comeca sem ruido e no for faz o calculo pra todos os ruidos nesse t
		# v_rmseK[0,t] = rmseKnn
		v_rmseS[0,t] = rmseSVD

		ruido_atual = porcentagem_ruido

		for iteracao in range(1, n_ruidos+1):
			print("  ruido ", ruido_atual,"%...\n")
			# d_treino_ruido = d_treino_t com ruido
			d_treino_ruido = flip_noise(d_treino_t, ruido_atual, tamanho_batch, t, indices_treino_t)
			# mae_ruidoKnn, rmse_ruidoKnn = knn.knn(d_treino_ruido, d_teste, n_usuarios, n_filmes, indices_teste, tipo_similaridade)
			mae_ruidoSVD, rmse_ruidoSVD = svd.svd(d_treino_ruido, d_teste, n_usuarios, n_filmes, indices_teste, indices_treino_t)

			# v_rmseK[iteracao,t] = rmse_ruidoKnn
			v_rmseS[iteracao,t] = rmse_ruidoSVD

			if ruido_atual == 25:
				ruido_atual += 25
			else:
				ruido_atual += incremento_ruido
		t +=1

		if t < n_batches:
			# atualiza o d_treino_t(treino baseado no tempo), que agora eh o anterior +1 batch de ratings
			# d_treino_t, indices_treino_t, i_rating_usuarios, usuario_atual = atualizar_treino_t(d_treino, n_batches, tamanho_batch, n_usuarios, n_filmes, d_treino_t, indices_treino_t, i_rating_usuarios, usuario_atual, tamanho_treino, t)
			d_treino_t, indices_treino_t, indices_timestamp_atual = atualizar_treino_timestamp(d_treino, n_batches, tamanho_batch, d_treino_t, indices_treino_t, tamanho_treino, t, timestamp_ord, indices_timestamp_atual)
	print("plotando...")

	plots = []
	ruido_atual = porcentagem_ruido
	x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	# x = [1, 2,3,4]
	#
	# for i in range(n_ruidos+1):
	# 	plot = plt.subplot()
	# 	# print(v_rmseK[i, :])
	# 	if i == 0:
	# 		plot.plot(x,v_rmseK[i,:],label='ruido 0' )
	# 	else:
	# 		plot.plot(x,v_rmseK[i,:],label='ruido {v}'.format(v=str(ruido_atual) ) )
	#
	# 		if ruido_atual == 25:
	# 			ruido_atual += 25
	# 		else:
	# 			ruido_atual += incremento_ruido
	# 	plot.legend()
	# 	plots.append(plot)


	labels = ['10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']
	# labels = ['10%', '20%','30%', '40%']
	# # labels = ['0%', '25%', '50%', '75%', '100%']
	# plt.xticks(x, labels, rotation='horizontal')
	# plt.title("KNN")
	# # plt.xlim(-0.2, 1.0)
	# plt.xlabel("Tempo")
	# plt.ylabel("RMSE")
	# plt.grid(True)
	# plt.show()


	ruido_atual = porcentagem_ruido

	for i in range(n_ruidos+1):
		plotS = plt.subplot()
		if i == 0:
			plotS.plot(x, v_rmseS[i,:],label='ruido 0' )
		else:
			plotS.plot(x, v_rmseS[i,:],label='ruido {v}'.format(v=str(ruido_atual) ) )
			if ruido_atual == 25:
				ruido_atual += 25
			else:
				ruido_atual += incremento_ruido
		plotS.legend()
		plots.append(plotS)

	plt.xticks(x, labels, rotation='horizontal')
	plt.title("RSVD")
	plt.xlabel("Tempo")
	plt.ylabel("RMSE")
	plt.grid(True)
	plt.show()




if __name__ == '__main__':
	main()
