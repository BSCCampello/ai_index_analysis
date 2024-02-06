import numpy as np
import functions
import pandas as pd
import sys
from tabulate import tabulate
import itertools
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import condorcet
np.set_printoptions(suppress=True)
from itertools import chain
from kendal_tau import Kendal
# Em 2023 a ordem de preferência dos criterios, baseado no peso que eles dão, é: research (26%), commercial vent (24%), talent (15%), dev (14%), infra (11%), op. env. (6%), gov stra. (4%)
# Nos dados do excel, as colunas estão na ordem: infra, op. env., talent, dev, research, commercial vent, gov stra.
# Assim, para 2023, considerando 0 o criterio menos preferível e 6 o mais preferivel, o vetor: [2, 1, 4, 3, 6, 5, 0], indica a ordem que os pesos devem ser gerados no smaa



# MODEL PARAMETER DEFINITION:

# q_column define quantas colunas do resultado vou gerar para a tabela do artigo, no BRACIS usamos os 6 primeiros paises
q_column = 12

# numInterationSmaa define quantas vezes rodamos a simulação, no artigo do BRACIS foram 10000
numInterationSmaa = 10000

# Decidir se LER pesos de acordo com uma ordem de prefências. usar True caso queira ler e executar o arquivo de pesos com ordem de preferência. Caso seja totalmente aleatório, usar False
ranking_with_criteria_preferences = True

# False if just READ file from computer, True if new weights should be generate
weight_file_name = "data/weight_vector.csv"
weight_file_name_ordered = "data/weight_vector_ordered.csv"


file_name = 'data/data_socores_2023_used_in_the_paper.xlsx'
#file_name = 'data.xlsx'
numCriteria = 7




vect_color_map = ["coolwarm"]

if ranking_with_criteria_preferences:
    print('Reading oredered weights')
    weight_file_name = weight_file_name_ordered
else:
    print('Reading random weights')

with open(weight_file_name, 'r') as arquivo_csv:
    leitor_csv = csv.reader(arquivo_csv)
    pesos_lista = []
    for linha in leitor_csv:
        pesos_lista.append(linha)
    w_completo = np.array(pesos_lista, dtype=float)


# The function read_data_from_excel() is to read the data, and it will return an ordered dictionary with countries names as keys and an array of criteria values as dictionary values.
data_dict = functions.read_data_from_excel(file_name)

# decMatrix = row alternatives (countries), columns 7 criteria (Talent, Infrastructure, Operating Environment, Research, Development, Government Strategy, Commercial)
# Create a decision matrix from the dictionary data.
DecisionMatrix = functions.decMatrix(data_dict)




# START THE SMAA METHOD:

# Create a matrix of zeros with the number of rows and columns equal to the number of alternatives (Countries). This matrix represents the probability of each alternative being in each position.
prob_matrix = np.zeros((len(DecisionMatrix), len(DecisionMatrix)))

# SMAA:

vetor_todos_rankings = []
for i, w in enumerate(w_completo):

    # Weight sum method:
    score = np.dot(DecisionMatrix, w)

    ranking = functions.sort_order(score)

    vetor_todos_rankings.append(ranking)

    for index, position in enumerate(ranking):

        prob_matrix[index][position] += 1




prob_matrix = np.array(prob_matrix).T
df_todos_os_rankings = pd.DataFrame(vetor_todos_rankings)
ranking_condorcet = condorcet.compute_ranks(df_todos_os_rankings)
print(ranking_condorcet)





# RELATED TO smaa matrix probability
percent_alt_positions = np.array(prob_matrix / numInterationSmaa * 100)
# Create a DataFrame from the array
df_percent_alt_positions = pd.DataFrame(percent_alt_positions)
print(df_percent_alt_positions)
sns.color_palette("mako", as_cmap=True)





# Plot the heatmap using the Seaborn heatmap function
for color_map in vect_color_map:
    file_name1 = 'data/heatmap_tortoise_%s_%s.png'%(color_map, ranking_with_criteria_preferences)
    functions.gerar_grafico_heatmap(df_percent_alt_positions, color_map, file_name1)



    #aquiu
    # Combine o vetor e a matriz em uma lista de tuplas
    combined = list(zip(ranking_condorcet, prob_matrix))

    # Ordene a lista com base no vetor (ordem decrescente)
    sorted_combined = sorted(combined, key=lambda x: x[0][0])

    # Separe o vetor ordenado e a matriz ordenada
    ranking_condorcet_ordenado, matriz_ordenada = zip(*sorted_combined)

    # Converta o vetor e a matriz ordenada de volta para listas
    ranking_condorcet_ordenado = list(ranking_condorcet_ordenado)
    prob_matriz_ordenada_condorcet = list(matriz_ordenada)
    soma_diagonal = np.trace(prob_matriz_ordenada_condorcet)
    print('soma diagonal condorcet')
    print(soma_diagonal)

    soma_diagonal = np.trace(prob_matrix)

    print('soma diagonal tortoise')
    print(soma_diagonal)
    #acabou


    file_name2 = 'data/heatmap_weight_sum_condorcet_%s_%s.png' % (color_map, ranking_with_criteria_preferences)
    functions.gerar_grafico_heatmap(prob_matriz_ordenada_condorcet, color_map, file_name2)



#ranking_condorcet_plano = list(chain.from_iterable(ranking_condorcet))
#print(ranking_condorcet_plano)


tau_condorcet = []
tau_tortoise = []
ranking_tortoise = list(range(62))





for ranking_smaa in vetor_todos_rankings:
    kendal_condorcete_obj_order = Kendal()
    kendal_condorcet_order = kendal_condorcete_obj_order.run(ranking_smaa, ranking_condorcet)
    tau_condorcet.append(kendal_condorcet_order)

    kendal_tortoise_obj_order = Kendal()
    kendal_tortoise_order = kendal_tortoise_obj_order.run(ranking_smaa, ranking_tortoise)
    tau_tortoise.append(kendal_tortoise_order)

# Calcular média e mediana
media_condorcet = np.mean(tau_condorcet)
mediana_condorcet = np.median(tau_condorcet)

media_tortoise = np.mean(tau_tortoise)
mediana_tortoise = np.median(tau_tortoise)

# Imprimir resultados
print("Média Tau Condorcet:", media_condorcet)
print("Mediana Tau Condorcet:", mediana_condorcet)

print("Média Tau Tortoise:", media_tortoise)
print("Mediana Tau Tortoise:", mediana_tortoise)

# Criar um boxplot
plt.boxplot([tau_condorcet, tau_tortoise], labels=['Tau Condorcet', 'Tau Tortoise'])

# Adicionar rótulos e título
plt.xlabel('Métodos')
plt.ylabel('Valores Tau')
plt.title('Boxplot de Tau para Condorcet e Tortoise')

# Exibir o boxplot
plt.show()



dict_percent_alt_positions = {}
for chave in data_dict.keys():
    dict_percent_alt_positions[chave] = percent_alt_positions[:, list(data_dict.keys()).index(chave)]


# Converter o dicionário em uma lista de listas
lista_de_listas = [list(coluna) for coluna in itertools.islice(dict_percent_alt_positions.values(), 0, 10)]




# Obter as primeiras 10 colunas e 10 linhas
the_first_q_columns = [linha[:q_column] for linha in lista_de_listas[:q_column]]



# Criar um DataFrame a partir das primeiras 10 colunas e 10 linhas
df = pd.DataFrame(the_first_q_columns, columns=list(dict_percent_alt_positions.keys())[:q_column])


df_transposto = df.T

# imprima a tabela LaTeX
print("matriz de probabilidades")
print(df_transposto.to_latex())











