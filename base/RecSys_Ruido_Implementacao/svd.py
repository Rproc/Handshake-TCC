# coding: utf-8
import collections as ct
import numpy as np
import random
import math
import time
from copy import copy


def MAE(dataset, previsto, n_usuarios, n_filmes):
	soma_erro = 0.0
	n = 0
	for i in range(n_usuarios):
		for j in range(n_filmes):
			if dataset[i,j] > 0.1:
				soma_erro += abs(dataset[i,j] - previsto[i,j])
				n += 1
	return soma_erro/n


def RMSE(dataset, previsto, n_usuarios, n_filmes):
	soma_erro = 0.0
	n = 0
	for i in range(n_usuarios):
		for j in range(n_filmes):
			if dataset[i,j] > 0.1:
				soma_erro += pow(dataset[i,j] - previsto[i,j],2)
				n += 1
	return math.sqrt(soma_erro/n)


def prever(indices_teste,q,p,n_usuarios,n_filmes, media_global, bias_i, bias_u):
	previsto = np.zeros((n_usuarios,n_filmes), dtype=np.float)

	for tupla in indices_teste:
		u = tupla[0]
		i = tupla[1]
		previsto[u,i] = limitar(media_global + bias_u[u] + bias_i[i] + np.dot(q[i,:], p[u,:]))

	return previsto

def regularized_squared_error(d_treino, q, p, indices_treino, lamb, media_global, bias_i, bias_u):
	erro = 0.0
	for tupla in indices_treino:
		u = tupla[0]
		i = tupla[1]
		previsto = limitar(media_global + bias_u[u] + bias_i[i] + np.dot(q[i,:], p[u,:]))
		erro = pow( (d_treino[u,i] - previsto ), 2 ) + lamb*(np.sum(q[i,:]**2) + np.sum(p[u,:]**2) + bias_u[u]**2 + bias_i[i]**2)

	return erro

def calcular_media_global(dataset,n_usuarios,n_filmes):
	qtd = 0
	soma = 0
	for i in range(n_usuarios):
		for j in range(n_filmes):
			soma += dataset[i,j]
			if dataset[i,j] > 0: qtd += 1
	return soma/qtd

def calcular_media_filmes(dataset,n_usuarios,n_filmes,media_global,alpha):
	media_filmes = np.zeros(n_filmes,dtype=np.float)
	bias_i = np.zeros(n_filmes,dtype=np.float)
	for j in range(n_filmes):
		qtd = 0
		soma = 0
		for i in range(n_usuarios):
			soma += float(dataset[i,j])
			if dataset[i,j] > 0: qtd += 1
		try:
			bi = soma/qtd
		except:
			bi = 0
		bias_i[j] = bi
		media_filmes[j] = shrink(bi, qtd, alpha, media_global)
	return media_filmes

def calcular_media_usuarios(dataset,n_usuarios,n_filmes,media_global,alpha):
	media_usuarios = np.zeros(n_usuarios,dtype=np.float)
	bias_u = np.zeros(n_usuarios,dtype=np.float)
	for i in range(n_usuarios):
		qtd = 0
		soma = 0
		for j in range(n_filmes):
			soma += float(dataset[i,j])
			if dataset[i,j] > 0: qtd += 1
		try:
			bu = soma/qtd
		except:
			bu = 0
		bias_u[i] = bu
		media_usuarios[i] = shrink(bu, qtd, alpha, media_global)
		# media_filmes[j] = ( (alpha / (alpha + qtd)) * media_global) + (qtd / (alpha + qtd)) * bi
	return media_usuarios

def calcular_bias_filmes(dataset,n_usuarios,n_filmes,media_global,alpha, media_usuarios):
	bias_i = np.zeros(n_filmes,dtype=np.float)
	for j in range(n_filmes):
		qtd = 0
		desvio = 0
		for i in range(n_usuarios):
			if dataset[i,j] > 0:
				qtd += 1
				desvio += float(dataset[i,j] - media_usuarios[i])
		try:
			bi = desvio/qtd
		except:
			bi = 0
		bias_i[j] = bi

	return bias_i

def calcular_bias_filmes_std(dataset,n_usuarios,n_filmes,media_global,alpha):
	bias_i = np.zeros(n_filmes,dtype=np.float)
	for j in range(n_filmes):
		qtd = 0
		desvio = 0
		for i in range(n_usuarios):
			if dataset[i,j] > 0:
				qtd += 1
				desvio += float(dataset[i,j] - media_global)
		try:
			bi = desvio/qtd
		except:
			bi = 0
		bias_i[j] = bi

	return bias_i

def calcular_bias_usuarios(dataset,n_usuarios,n_filmes,media_global,alpha, media_filmes):
	bias_u = np.zeros(n_usuarios,dtype=np.float)
	for i in range(n_usuarios):
		qtd = 0
		desvio = 0
		for j in range(n_filmes):
			if dataset[i,j] > 0:
				qtd += 1
				desvio += float(dataset[i,j] - media_filmes[j])
		try:
			bu = desvio/qtd
		except:
			bu = 0
		bias_u[i] = bu

	return bias_u

def calcular_bias_usuarios_std(dataset,n_usuarios,n_filmes,media_global,alpha):
	bias_u = np.zeros(n_usuarios,dtype=np.float)
	for i in range(n_usuarios):
		qtd = 0
		desvio = 0
		for j in range(n_filmes):
			if dataset[i,j] > 0:
				qtd += 1
				desvio += float(dataset[i,j] - media_global)
		try:
			bu = desvio/qtd
		except:
			bu = 0
		bias_u[i] = bu

	return bias_u

def shrink(nota, qtd, alpha, media_global):
	shrunk = ( (alpha / (alpha + qtd)) * media_global) + (qtd / (alpha + qtd)) * nota
	if(shrunk < 1):
		shrunk = 1
	elif(shrunk > 5):
		shrunk = 5
	return shrunk

def limitar(nota):
	if(nota < 1):
		nota = 1
	elif(nota > 5):
		nota = 5
	return nota

def svd(d_treino, d_teste,n_usuarios, n_filmes, indices_teste, indices_treino):
	# n_usuarios = 943
	# n_filmes = 1682
	porcentagem_treino = 90
	iteracao = 0
	k = 50
	lamb = 0.02
	l_rate = 0.005
	limite_erro = 0.0001
	diferenca = np.inf
	n_piora = 0
	max_n_piora = 10
	max_iteracoes = 100
	alpha = 1
	count = 1
	contador = 0

	# texto2 = texto
	mae = 0
	rmse = 0
	mae_mean = 0
	rmse_mean = 0

	while contador < count:

		# dataset, d_treino, d_teste, indices_treino, indices_teste = criar_datasets(n_usuarios,n_filmes,porcentagem_treino)
		media_global = calcular_media_global(d_treino,n_usuarios,n_filmes)
		media_filmes = calcular_media_filmes(d_treino,n_usuarios,n_filmes,media_global, alpha)
		media_usuarios = calcular_media_usuarios(d_treino,n_usuarios,n_filmes,media_global, alpha)


		bias_i = calcular_bias_filmes(d_treino,n_usuarios,n_filmes,media_global, alpha, media_usuarios)
		bias_u = calcular_bias_usuarios(d_treino,n_usuarios,n_filmes,media_global, alpha, media_filmes)

		# bias_i = np.zeros(n_filmes)
		# bias_u = np.zeros(n_usuarios)
		#
		# bias_i = calcular_bias_filmes_std(d_treino,n_usuarios,n_filmes,media_global, alpha)
		# bias_u = calcular_bias_usuarios_std(d_treino,n_usuarios,n_filmes,media_global, alpha)

		U, s, v = np.linalg.svd(d_treino)
		p = U[:,0:k]
		q = v[:,0:k]


		# aqui comeca
		erro_min = regularized_squared_error(d_treino, q, p, indices_treino, lamb, media_global, bias_i, bias_u)
		erro_atual = np.inf
		best_q = copy(q)
		best_p = copy(p)
		# print("calibrando...")
		t0 = time.clock()


		while n_piora < max_n_piora and iteracao < max_iteracoes:

			rand_index = np.random.permutation(len(indices_treino))
			# for tupla in indices_treino:
			for rand in rand_index:
				tupla = indices_treino[rand]
				u = tupla[0]
				i = tupla[1]

				previsto = limitar(media_global + bias_u[u] + bias_i[i] + np.dot(q[i,:], p[u,:]))

				erro_ui = d_treino[u,i] - previsto
				# salto gigante?
				aux = copy(q[i,:])
				q[i,:] = q[i,:] + l_rate*(erro_ui*p[u,:] - lamb*q[i,:])
				p[u,:] = p[u,:] + l_rate*(erro_ui*aux - lamb*p[u,:])
				bias_u[u] = bias_u[u] + l_rate*(erro_ui - lamb*bias_u[u])
				bias_i[i] = bias_i[i] + l_rate*(erro_ui - lamb*bias_i[i])

				# p[u,:] = p[u,:] + l_rate*(erro_ui*q[i,:] - lamb*p[u,:])

				# for feature in range(k):
				# 	q[i,k] = q[i,k] + l_rate*(erro_ui*p[u,:] - lamb*q[i,:])
				# 	p[u,k] = p[u,k] + l_rate*(erro_ui*q[i,:] - lamb*p[u,:])

			erro_atual = regularized_squared_error(d_treino, q, p, indices_treino, lamb, media_global, bias_i, bias_u)
			diferenca = abs(erro_atual - erro_min)

			if erro_atual < erro_min:
				erro_min = erro_atual
				best_q = copy(q)
				best_p = copy(p)

				if diferenca > limite_erro:
					n_piora = 0
				else:
					n_piora += 1
			else:
				n_piora += 1

			if n_piora == 5 and l_rate >= 0.001:
				l_rate *= 0.5

			iteracao += 1

		print("Calibrou. Tempo: ", time.clock() - t0)

		# print("\nPrevendo...")
		t0 = time.clock()
		previsto = prever(indices_teste,best_q,best_p,n_usuarios,n_filmes, media_global, bias_i, bias_u)
		# print("Previu. Tempo: ", time.clock() - t0)

		mae = MAE(d_teste,previsto,n_usuarios,n_filmes)
		rmse = RMSE(d_teste,previsto,n_usuarios,n_filmes)
		print("\nMAE: ", mae)
		print("RMSE: ", rmse)

		mae_mean += mae
		rmse_mean += rmse


		iteracao = 0
		n_piora = 0
		bias_i = np.zeros(n_filmes)
		bias_u = np.zeros(n_usuarios)
		l_rate = 0.01


		contador+=1

	# mae_mean = mae_mean/count
	# rmse_mean = rmse_mean/count

	return mae_mean, rmse_mean
