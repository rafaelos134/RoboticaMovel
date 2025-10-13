import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time

import networkx as nx

def mapa(image_caminho,start_node,end_node,escala_imagem,tamanho_cell):

    fig = plt.figure(figsize=(8,8), dpi=100)
    ax = fig.add_subplot(111, aspect='equal')

    # Invertendo os valores para visualização (Branco - 0, Preto - 1)

    img = mpimg.imread(image_caminho).astype(float)

    if img.ndim == 3:
        img = img[:, :, 0]

    img = 1 - img

    # Apenas para garantir que só teremos esses dois valores
    threshold = 0.5
    img[img > threshold] = 1
    img[img<= threshold] = 0

    # Dimensões do mapa informado em metros (X, Y)
    map_dims = np.array(escala_imagem)

    # Escala Pixel/Metro
    sy, sx = img.shape[:2] / map_dims

    # Tamanho da célula do nosso Grid (em metros)
    cell_size = tamanho_cell

    rows, cols = (map_dims / cell_size).astype(int)
    grid = np.zeros((rows, cols))

    # Preenchendo o Grid
    # Cada célula recebe o somatório dos valores dos Pixels
    for r in range(rows):
        for c in range(cols):
            
            xi = int(c*cell_size*sx)
            xf = int(xi + cell_size*sx)
            
            yi = int(r*cell_size*sy)
            yf = int(yi + cell_size*sy)
                        
            grid[r, c] = np.sum(img[yi:yf,xi:xf])
            
    # Binarizando as células como ocupadas (1) ou Não-ocupadas (0)       
    grid[grid > threshold] = 1
    grid[grid<= threshold] = 0   

    # Criando o Grafo para o Grid

    # Criando vértices em todas as células
    G = nx.grid_2d_graph(rows, cols) 
    G_complete = G.copy()

    # Removendo células que estão em células marcas com obstáculos
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:  
                G.remove_node((r,c))

    # Os vértices serão plotados no centro da célula  
    pos = {node:(node[1]*cell_size+cell_size/2, map_dims[0]-node[0]*cell_size-cell_size/2) for node in G.nodes()}

    # Mapa
    obj = ax.imshow(grid, cmap='Greys', extent=(0, map_dims[1], 0,map_dims[0]))

    # Caminho
    path = nx.shortest_path(G, source=start_node, target=end_node)
    nx.draw_networkx_nodes(G, pos, nodelist=path, node_size=100, node_color='b')

    return pos, path

import os

print(os.getcwd())


pos, path = mapa('src/mapas_meus/mapa1_invertido_extend.png',(11,11),(79,80),(100),(1))


