import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math 
import networkx as nx
import numpy as np
from skimage.draw import line
from matplotlib.colors import ListedColormap

from coppeliasim_zmqremoteapi_client import RemoteAPIClient 

def Rz(theta):
  
    return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                      [ np.sin(theta), np.cos(theta) , 0 ],
                      [ 0            , 0             , 1 ]])

# debug do codigo
LOG_ODDS_MIN = -10.0
LOG_ODDS_MAX = 10.0

# --- PARÂMETROS DO MAPA ---
MAP_RESOLUTION = 0.1  # 10 cm por célula (problema quando muda a resolucao)
MAP_SIZE_X = 10.0      # metros
MAP_SIZE_Y = 10.0      # metros
MAP_ORIGIN_X = -5.0    # Canto inferior esquerdo X
MAP_ORIGIN_Y = -5.0    # Canto inferior esquerdo Y
nrows = int(MAP_SIZE_Y / MAP_RESOLUTION)
ncols = int(MAP_SIZE_X / MAP_RESOLUTION)
MAX_SCAN_DISTANCE = 5.0

# Valores de Log-Odds
l_occ = 0.9
l_free = -0.4
LOG_ODDS_MIN = -5.0
LOG_ODDS_MAX = 5.0


def world_to_grid_xy(wx, wy):
    
    # Converte mundo (x,y) [m] -> índice de célula (ix, iy)
    # origem está no centro do mapa 0,0
    
    ix = int((wx - MAP_ORIGIN_X) / MAP_RESOLUTION)
    iy = int((wy - MAP_ORIGIN_Y) / MAP_RESOLUTION)
    
    # ix = int((ncols / 2) + (wx / cell_size)) # parentezes modificado, caso tenha problema rever
    # iy = int((nrows / 2) + (wy / cell_size))

    if 0 <= ix < ncols and 0 <= iy < nrows:
        return ix, iy
    return None, None

MAX_RANGE_THRESHOLD = MAX_SCAN_DISTANCE * 0.99
# Calculo do log_odds (explicar no video)
def update_map_log_odds(log_odds_map, robot_pos, laser_data, l_occ = 0.9, l_free = -0.4):
    
    # define a posicao e rotacao do robo
    x, y, theta_r = robot_pos[0], robot_pos[1], robot_pos[2]

    # índice da célula do robô (coluna, linha)
    x_Grid, y_Grid = world_to_grid_xy(x, y)
    
    # caso o robo esteja fora do mapa
    if x_Grid is None or y_Grid is None:
        return log_odds_map

    for d, ang in laser_data: # verificar como esse laiser data está chegando
        # filtrar leituras inválidas
        if d <= 0.01: 
            continue

        global_ang = theta_r + ang
        
        posi_x = x + d * np.cos(global_ang)
        posi_y = y + d * np.sin(global_ang)

        posi_xGrid, posi_yGrid = world_to_grid_xy(posi_x, posi_y)
        
        if posi_xGrid is None or posi_yGrid is None:
            continue
        
        # Bresenham: linha de (iy_r, ix_r) até (iy, ix) -> rr (rows), cc (cols)
        rr, cc = line(y_Grid, x_Grid, posi_yGrid, posi_xGrid)
        
        # CASO 1: É um obstáculo REAL (distância < max)
        if d < MAX_RANGE_THRESHOLD:
            # Marcar células intermediárias como livres
            if len(rr) > 2:
                log_odds_map[rr[1:-1], cc[1:-1]] += l_free
            
            # Marcar célula de impacto como ocupada
            log_odds_map[posi_yGrid, posi_xGrid] += l_occ
        
        # CASO 2: É um feixe de MAX RANGE (sem obstáculo)
        else:
            # Marcar TODAS as células ao longo do feixe como livres
            # (Exceto a primeira, onde o robô está)
            if len(rr) > 1:
                log_odds_map[rr[1:], cc[1:]] += l_free

        # marcar a célula de impacto como ocupada
        log_odds_map[posi_yGrid, posi_xGrid] += l_occ

    # clamp para estabilidade numérica
    np.clip(log_odds_map, LOG_ODDS_MIN, LOG_ODDS_MAX, out=log_odds_map)
    return log_odds_map





# tenho que mexer aq
def parse_laser_data(raw):
    """
    Converte retorno do sensor para lista de (d, ang) (metros, radianos no frame do sensor).
    Ajuste aqui se seu Hokuyo retornar outro formato.
    """
    # Caso já seja uma lista de (d, ang):
    if isinstance(raw, (list, tuple)) and len(raw) and isinstance(raw[0], (list, tuple)):
        return [(float(d), float(a)) for d, a in raw]

    # Caso seja apenas uma lista/array de ranges
    if isinstance(raw, (list, np.ndarray)):
        ranges = np.array(raw, dtype=float)
        n = len(ranges)
        if n == 0:
            return []
        # suposição típica: campo de visão -pi/2..+pi/2
        angles = np.linspace(-np.pi/2, np.pi/2, n)
        return [(float(np.ravel(r)[0]), float(a)) for r, a in zip(ranges, angles)]


    # se vazio ou formato desconhecido
    return []





# -----------------------
# Setup CoppeliaSim (exemplo)
# -----------------------
client = RemoteAPIClient()
sim = client.require("sim")
sim.stopSimulation()
time.sleep(0.2)
sim.startSimulation()
sim.setStepping(True)

robotname = "kobuki"
robot = sim.getObject(f'/{robotname}')
l_wheel = sim.getObject(f'/{robotname}/wheel_left_drop_sensor/kobuki_leftMotor')
r_wheel = sim.getObject(f'/{robotname}/wheel_right_drop_sensor/kobuki_rightMotor')
hokuyo = HokuyoSensorSim(sim, f"/{robotname}/fastHokuyo")  # seu wrapper



# -----------------------
# Config mapa / constantes
# -----------------------
map_size_m = 10.0
cell_size = 0.1                      # m por célula
ncols = int(map_size_m / cell_size)  # cols = eixo X
nrows = int(map_size_m / cell_size)  # rows = eixo Y
grid_origin = np.array([-map_size_m/2, -map_size_m/2])  # world coord do canto inferior esquerdo da célula (0,0)


L = 0.381  
r = 0.0975 

hist = []            
laser_global = []    



# log-odds map: shape (nrows, ncols) -> index como [row, col] = [y, x]
log_odds_map = np.zeros((nrows, ncols))



# colormap: 0=desconhecido(gray), 1=livre(white), 2=ocupado(black)
cmap = ListedColormap(["gray", "white", "black"])

# -----------------------
# Loop principal (visualização em metros)
# -----------------------
plt.ion()
fig, ax = plt.subplots(figsize=(6,6))

robot_path_world = []  # lista de (x,y) world coordinates

for step in range(400):
    sim.step()
    # pos = sim.getObjectPosition(robot, -1)
    # ori = sim.getObjectOrientation(robot, -1)
    
    
    pos = sim.getObjectPosition(robot, sim.handle_world)
    ori = sim.getObjectOrientation(robot, sim.handle_world)
    
    
    hist.append([pos[0], pos[1]])
    
    laser_data = hokuyo.getSensorData() # Retorna [ângulo, dist]
    laser_global.extend(transform_laser_to_global(laser_data, pos, ori))

    robot_pos = [pos[0], pos[1], ori[2]]
    
    
    laser_data_swapped = [[d, a] for a, d in laser_data]
        
    log_odds_map = update_map_log_odds(
        log_odds_map, 
        robot_pos, 
        laser_data_swapped, # Usar os dados com ordem invertida
        l_occ, 
        l_free
    )

    log_odds_map = update_map_log_odds(log_odds_map, robot_pos, laser_data)

    # salvar rastro em coords do mundo
    robot_path_world.append((robot_pos[0], robot_pos[1]))
    
    wl, wr =  controle(laser_raw,r,L)
    sim.setJointTargetVelocity(l_wheel, wl)
    sim.setJointTargetVelocity(r_wheel, wr)

    if step % 50 == 0:
        print("hi")
        prob = 1.0 / (1.0 + np.exp(-log_odds_map))
        grid_vis = np.zeros_like(prob)
        grid_vis[prob < 0.3] = 1  # livre
        grid_vis[prob > 0.7] = 2  # ocupado

        # --- cálculo do extent ---
        nrows, ncols = grid_vis.shape
        extent = [grid_origin[0], grid_origin[0] + ncols * cell_size,
                    grid_origin[1], grid_origin[1] + nrows * cell_size]

        # --- Atualização do plot ---
        ax.clear()
        ax.imshow(grid_vis, origin='lower', cmap=cmap, extent=extent, vmin=0, vmax=2)

        # desenhar rastro em coordenadas do mundo
        if robot_path_world:
            xs, ys = zip(*robot_path_world)
            ax.plot(xs, ys, color='blue', linewidth=1)

        # desenhar pose do robô
        arrow_scale = cell_size * 5
        ax.arrow(robot_pos[0], robot_pos[1],
                    arrow_scale*np.cos(robot_pos[2]), arrow_scale*np.sin(robot_pos[2]),
                    head_width=cell_size*2, head_length=cell_size*2, fc='red', ec='red')

        
        
        ax.set_title(f"Mapa Parcial - passo {step}")
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_aspect('equal')
        plt.pause(0.001)
        
        

# parar sim
sim.stopSimulation()
print('Program ended')

# -----------------------------------------------------------------
# Plot 1: Scatter (o que você já tinha, gera laiser.png)
# (Este código é do seu original.py e está correto)
# -----------------------------------------------------------------
print("Gerando Plot 1 (Scatter)...")
fig1 = plt.figure(figsize=(8,8), dpi=100)
ax1 = fig1.add_subplot(111, aspect='equal')

# 'hist' é a variável do seu script original.py
x, y = zip(*hist)
ax1.plot(x, y, '--k', label="Trajetória")
ax1.plot(x[0], y[0], 'go', markersize=10, label="Início")
ax1.plot(x[-1], y[-1], 'ro', markersize=10, label="Fim")

# 'laser_global' é a variável do seu script original.py
if len(laser_global) > 0:
    lx, ly = zip(*laser_global)
    ax1.scatter(lx, ly, s=2, c='r', alpha=0.3, label="Laser")

ax1.legend()
ax1.set_xlabel("X [m]")
ax1.set_ylabel("Y [m]")
ax1.grid(True)
ax1.set_title("Plot de Pontos do Laser (Scatter)")
plt.show()


# -----------------------------------------------------------------
# Plot 2: Mapa de Grade (gera grid.png)
# (Este é o código corrigido e integrado)
# -----------------------------------------------------------------
print("Gerando Plot 2 (Mapa de Grade)...")

# Converter log-odds para probabilidade (sua fórmula está correta)
# 'log_odds_map' foi criada e atualizada no seu loop 'while'
prob = 1.0 / (1.0 + np.exp(-log_odds_map)) 

# Criar mapa visual (0=desconhecido, 1=livre, 2=ocupado)
grid_vis = np.zeros_like(prob, dtype=np.uint8) # 0 = desconhecido
grid_vis[prob < 0.45] = 1 # 1 = livre (branco)
grid_vis[prob > 0.55] = 2 # 2 = ocupado (preto)

# Definir um colormap: 0=cinza, 1=branco, 2=preto
# (Substitui a variável 'cmap' indefinida)
cmap = ListedColormap(['#808080', '#FFFFFF', '#000000'])

plt.ioff()
fig2, ax2 = plt.subplots(figsize=(7,7))

# Definir a 'extent' usando as variáveis de configuração do mapa
# (Substitui 'grid_origin', 'ncols', 'cell_size')
extent = [MAP_ORIGIN_X, MAP_ORIGIN_X + MAP_SIZE_X, 
          MAP_ORIGIN_Y, MAP_ORIGIN_Y + MAP_SIZE_Y]
          
ax2.imshow(grid_vis, origin='lower', cmap=cmap, extent=extent, vmin=0, vmax=2)

# Plotar a trajetória (usando 'hist' ao invés de 'robot_path_world')
if hist:
    xs, ys = zip(*hist)
    ax2.plot(xs, ys, color='blue', linewidth=1, label="Trajetória")
    ax2.plot(xs[-1], ys[-1], 'ro', markersize=5, label="Fim") # Ponto final

# (Removi ax2.arrow() pois 'robot_pos' não está disponível aqui,
# e o ponto final vermelho já marca a posição final)

ax2.set_title("Mapa Final Explorado")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_aspect('equal')
ax2.legend()
plt.show()


