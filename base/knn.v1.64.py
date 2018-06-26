# coding: utf-8
import collections as ct
import numpy as np
import random
import math
import time
from copy import deepcopy

COSSENO = 'cos'
EUCLIDIANA = 'euc'
MANHATTAN = 'man'
CANBERRA = 'can'

def criar_datasets(n_usuarios, n_filmes,porcentagem_treino):
	dataset = np.zeros((n_usuarios,n_filmes), dtype=np.float)
	d_treino = np.zeros((n_usuarios,n_filmes), dtype=np.float)
	d_teste = np.zeros((n_usuarios,n_filmes), dtype=np.float)
	indices_teste = []
	i_arranjo = 0
	

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
		else:
			d_teste[i_usuario, i_filme] = info[2]
			if(int(info[2]) > 0):
				indices_teste.append((i_usuario, i_filme))
		i_arranjo += 1

	return dataset, d_treino, d_teste, indices_teste


def calcula_similaridade(u1,u2,tipo):
	if tipo == 'cos':
		similaridade = sim_cos(u1,u2)
	elif tipo == 'euc':
		try:
			similaridade = (1/dist_euc(u1,u2))
		except:
			similaridade = 0
	elif tipo == 'man':
		try:
			similaridade = (1/dist_manhattan(u1,u2))
		except:
			similaridade = 0
	elif tipo == 'can':
		try:
		    similaridade = 1/dist_canberra(u1,u2)
		except:
		    similaridade = 0

	return similaridade


def dist_euc(u1,u2):
	return math.sqrt(np.sum(pow(u1-u2,2)))


def dist_manhattan(u1,u2):
	return np.sum(abs(u1-u2))

def dist_canberra(u1,u2):
    return np.sum( abs(u1-u2) ) / np.sum( abs(u1) + abs(u2) )


def norma(v):
	soma = 0
	for i in range(len(v)):
		soma += v[i]**2
	return math.sqrt(soma)


def sim_cos(u1,u2):
	try:
		resultado = float(np.sum(u1 * u2))/( math.sqrt(np.sum(u1**2)) * math.sqrt(np.sum(u2**2)) )
	except:
		resultado = 0
	return resultado


def imprime_matriz_2_casas(matriz):
	linhas = len(matriz)
	colunas = len(matriz[0,:])
	for i in range(linhas):
		for j in range(colunas):
			print('%.2f' % matriz[i,j],"\t",end="")
		print("")


def vizinhos_mais_proximos(d_treino, usuario, filme, similaridades, k, tipo, k_min):

	vizinhos = np.zeros((len(similaridades),2))
	ind_aux = 0

	for v in range(len(similaridades)):
		# se alguma nota foi data
		if usuario != v and d_treino[v,filme] > 0.1:
			# se a similaridade eh inf, nao foi calculada ainda
			if np.isinf(similaridades[usuario,v]):
				similaridades[usuario,v] = calcula_similaridade(d_treino[usuario,:], d_treino[v,:], tipo)
				similaridades[v,usuario] = similaridades[usuario,v]
			vizinhos[ind_aux,0] = v
			vizinhos[ind_aux,1] = similaridades[usuario,v]
			ind_aux +=1

	# indices dos vizinhos mais proximos ordenados por similaridade
	ind_ordenados = np.argsort(vizinhos[0:ind_aux,1])
	ind_ordenados = list(reversed(ind_ordenados))

	if len(ind_ordenados) >= k_min:
		return vizinhos[ind_ordenados[0:k],0].astype(int)
	else:
		return None


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


def calcular_media_global(dataset,n_usuarios,n_filmes):
	qtd = 0
	soma = 0
	for i in range(n_usuarios):
		for j in range(n_filmes):
			soma += dataset[i,j]
			if dataset[i,j] > 0: qtd += 1
	return soma/qtd


def calcular_media_usuarios(dataset,n_usuarios,n_filmes,media_global,alpha):
	media_usuarios = np.zeros(n_usuarios,dtype=np.float)

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

		media_usuarios[i] = shrink(bu, qtd, alpha, media_global)
	return media_usuarios


# nota media de cada filme.
def calcular_media_filmes(dataset,n_usuarios,n_filmes,media_global,alpha):
	media_filmes = np.zeros(n_filmes,dtype=np.float)

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
		media_filmes[j] = shrink(bi, qtd, alpha, media_global)
		# media_filmes[j] = ( (alpha / (alpha + qtd)) * media_global) + (qtd / (alpha + qtd)) * bi
	return media_filmes


def prever(d_treino, d_teste, n_usuarios, n_filmes,similaridades,k, media_global, media_filmes, tipo, k_min, alpha, indices_teste, media_usuarios):
	previsto = np.zeros((n_usuarios,n_filmes), dtype=np.float)
	# for usuario in range(n_usuarios):
		# for filme in range(n_filmes):
	for tupla in indices_teste:
		usuario = tupla[0]
		filme = tupla[1]
		vizinhos = vizinhos_mais_proximos(d_treino,usuario,filme,similaridades,k,tipo, k_min)

		if vizinhos is not None:
			previsto[usuario,filme] = prever_nota(d_treino, usuario, filme, vizinhos, media_global, alpha, media_filmes, similaridades, media_usuarios)
		else:
			# print("NANI???")
			# previsto[usuario,filme] = shrink(media_filmes[filme,0], media_filmes[filme,1], alpha, media_global)
			previsto[usuario,filme] = media_filmes[filme]
	return previsto


def prever_nota(d_treino, usuario, filme, vizinhos, media_global, alpha,media_filmes,similaridades, media_usuarios):
	desvio = 0.0
	num = 0.0
	den = 0.0
	n = 0
	for v in vizinhos:
		if d_treino[v,filme] > 0.1:
			num += similaridades[usuario,v]*(d_treino[v,filme] - media_usuarios[v])
			den += similaridades[usuario,v]

	desvio = num/den
	nota = media_usuarios[usuario] + desvio

	if(nota < 1):
		nota = 1
	elif(nota > 5):
		nota = 5

	# return round(nota_shrunk, 4)
	return round(nota, 4)

def shrink(nota, qtd, alpha, media_global):
	shrunk = ( (alpha / (alpha + qtd)) * media_global) + (qtd / (alpha + qtd)) * nota
	if(shrunk < 1):
		shrunk = 1
	elif(shrunk > 5):
		shrunk = 5
	return shrunk

# similaridades: cosseno, Pearson, Spearman Kendall
def main():
	# dataset original:
	n_usuarios = 943
	n_filmes = 1682

	count = 0
	rmse_mean = 0
	mae_mean = 0
	k = 20

	k_min = 2
	alpha = 1
	tipo_similaridade = COSSENO
	porcentagem_treino = 90

	# parametros do experimento
	k_final = 6
	n_vezes = 10

	while k_min < k_final:
		mae_mean = 0
		rmse_mean = 0

		while count < n_vezes:

			# print("Criou datasets")

			# dataset de teste (fake.data)
			# n_usuarios = 22
			# n_filmes = 19

			dataset, d_treino, d_teste, indices_teste = criar_datasets(n_usuarios,n_filmes,porcentagem_treino)

			media_global = calcular_media_global(d_treino,n_usuarios,n_filmes)
			similaridades = np.full((n_usuarios,n_usuarios),np.inf,dtype=float)
			t0 = time.clock()

			media_filmes = calcular_media_filmes(d_treino,n_usuarios,n_filmes,media_global, alpha)
			media_usuarios = calcular_media_usuarios(d_treino,n_usuarios,n_filmes,media_global, alpha)
			# print("Calculou medias. tempo: ",time.clock() - t0)

			t0 = time.clock()
			previsto = prever(d_treino, d_teste, n_usuarios, n_filmes,similaridades,k, media_global, media_filmes, tipo_similaridade, k_min, alpha, indices_teste, media_usuarios)
			# print("previu. tempo: ",time.clock() - t0)

			# print("k: ", k, " Similaridade: ", tipo_similaridade)
			mae = MAE(d_teste,previsto,n_usuarios,n_filmes)
			mae_mean += mae

			# print("MAE: ", mae)

			rmse = RMSE(d_teste,previsto,n_usuarios,n_filmes)
			# print('rmse: ',rmse)
			rmse_mean += rmse
			# print("RMSE: ", rmse)

			count+=1

		count = 0

		print("K_min: ", k_min)
		print("Media do MAE: ", mae_mean/n_vezes)
		print("Media do RMSE: ", rmse_mean/n_vezes)

		k_min += 1


if __name__ == '__main__':
	main()
