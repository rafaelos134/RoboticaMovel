import numpy as np
# import matplotlib          # <-- 1. Importe o matplotlib "principal"
# matplotlib.use("Qt5Agg")  # <-- 2. Defina o backend para TkAgg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import math 
import networkx as nx
import numpy as np
from skimage.draw import line
from matplotlib.colors import ListedColormap

import HokuyoSensorSim
import controleWallFollowing

from coppeliasim_zmqremoteapi_client import RemoteAPIClient 

def Rz(theta):
  
    return np.array([[ np.cos(theta), -np.sin(theta), 0 ],
                      [ np.sin(theta), np.cos(theta) , 0 ],
                      [ 0            , 0             , 1 ]])


def pixel_to_world(x_px, y_px, img_size=64, world_size=10):
        scale = world_size / img_size
        x_world = ((x_px - img_size/2) * scale)
        y_world = (-(y_px - img_size/2) * scale) 
        return x_world, y_world

# mexendo atualmente
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
# l_occ = 0.9
# l_free = -0.4
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
def update_map_log_odds(log_odds_map, robot_pos, laser_data, step, last_seen,
                        l_occ=0.2, l_free=-0.7):
    
    x, y, theta_r = robot_pos[0], robot_pos[1], robot_pos[2]
    x_Grid, y_Grid = world_to_grid_xy(x, y)
    
    if x_Grid is None or y_Grid is None:
        return log_odds_map, last_seen

    # Limiar para considerar uma célula como "fortemente ocupada"
    # (Não apague células acima deste valor)
    L_OCC_THRESHOLD = 3 
    L_FREE_THRESHOLD = -3
    
    
    for d, ang in laser_data:
        if d <= 0.01: 
            continue

        global_ang = theta_r + ang
        posi_x = x + d * np.cos(global_ang)
        posi_y = y + d * np.sin(global_ang)
        posi_xGrid, posi_yGrid = world_to_grid_xy(posi_x, posi_y)
        
        if posi_xGrid is None or posi_yGrid is None:
            continue
        
        rr, cc = line(y_Grid, x_Grid, posi_yGrid, posi_xGrid)
        
        if len(rr) <= 1:
            continue

        # --- LÓGICA DE ATUALIZAÇÃO CORRIGIDA ---

        # 1. Obter todas as células do feixe (exceto a do robô)
        beam_rows, beam_cols = rr[1:], cc[1:]

        # 2. Aplicar 'livre' (l_free) ao longo do feixe, EXCETO no ponto final
        if len(beam_rows) > 1:
            path_rows, path_cols = beam_rows[:-1], beam_cols[:-1]
            
            # Verificar o estado atual dessas células
            current_log_odds = log_odds_map[path_rows, path_cols]
            
            # Criar uma máscara de células que NÃO estão fortemente ocupadas
            free_or_unknown_mask = current_log_odds < L_OCC_THRESHOLD
            
            # Aplicar 'l_free' APENAS a essas células
            rows_to_free = path_rows[free_or_unknown_mask]
            cols_to_free = path_cols[free_or_unknown_mask]
            log_odds_map[rows_to_free, cols_to_free] += l_free

        # 3. Tratar o PONTO FINAL (a última célula do feixe)
        end_row, end_col = beam_rows[-1], beam_cols[-1]
        
        if d < MAX_RANGE_THRESHOLD:
            # if log_odds_map[end_row, end_col] > L_FREE_THRESHOLD:
                log_odds_map[end_row, end_col] += l_occ
        else:
            # Feixe de Max Range -> Marcar ponto final como LIVRE
            # (Também checando para não apagar um obstáculo)
            # if log_odds_map[end_row, end_col] < L_OCC_THRESHOLD:
                log_odds_map[end_row, end_col] += l_free

        # 4. Atualizar o tempo de visão de todo o feixe
        last_seen[rr, cc] = step

    # Clamp para estabilidade
    np.clip(log_odds_map, LOG_ODDS_MIN, LOG_ODDS_MAX, out=log_odds_map)
    return log_odds_map, last_seen


plt.ion() # Ligar modo interativo
# Criar uma figura com 2 subplots (1 linha, 2 colunas)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
fig.suptitle("Mapeamento e Trajetória em Tempo Real", fontsize=16)
cmap_vis = ListedColormap(['#808080', '#FFFFFF', '#000000'])
N_LASER_POINTS_PLOT = 1000 # Limite de pontos de laser para plotar (performance)
map_extent = [MAP_ORIGIN_X, MAP_ORIGIN_X + MAP_SIZE_X, 
              MAP_ORIGIN_Y, MAP_ORIGIN_Y + MAP_SIZE_Y]


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
hokuyo = HokuyoSensorSim.HokuyoSensorSim(sim, f"/{robotname}/fastHokuyo")  # seu wrapper

# funcoes a verificar
world = pixel_to_world(30,34,100,10)
goal_position = np.array([world[0], world[1]])

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
last_seen = np.zeros_like(log_odds_map)


# colormap: 0=desconhecido(gray), 1=livre(white), 2=ocupado(black)
cmap = ListedColormap(["gray", "white", "black"])

# -----------------------
# Loop principal (visualização em metros)
# -----------------------
plt.ion()
fig, ax = plt.subplots(figsize=(6,6))

robot_path_world = []  # lista de (x,y) world coordinates

LIMITE_TEMPO_SIM = 20 * 60
step = 0

while sim.getSimulationState() != sim.simulation_stopped:
    current_sim_time = sim.getSimulationTime()

    sim.step()

    step +=1

    
    if step > 1000:
        print(f"Tempo limite de {LIMITE_TEMPO_SIM}s atingido. Parando a simulação.")
        break

    pos = sim.getObjectPosition(robot, sim.handle_world)
    ori = sim.getObjectOrientation(robot, sim.handle_world)
    
    
    hist.append([pos[0], pos[1]])
    
    laser_data = hokuyo.getSensorData() # Retorna [ângulo, dist]
    laser_global.extend(HokuyoSensorSim.transform_laser_to_global(laser_data, pos, ori))

    robot_pos = [pos[0], pos[1], ori[2]]
    
    
    laser_data_swapped = [[d, a] for a, d in laser_data]
        
    log_odds_map, last_seen = update_map_log_odds(
        log_odds_map,
        robot_pos,
        laser_data_swapped,
        step,
        last_seen
    )
    
    # if step % 10 == 0:
    #     unseen_mask = (step - last_seen) > 400
    #     log_odds_map[unseen_mask] *= 0.995

    # log_odds_map = update_map_log_odds(log_odds_map, robot_pos, laser_data)

    # salvar rastro em coords do mundo
    robot_path_world.append((robot_pos[0], robot_pos[1]))
    wl, wr = controleWallFollowing.controle_wall_following_com_escape(sim, robot, laser_data, r, L)
    
    # wl, wr =  controle_campos(world, goal_position, np.array([pos[0], pos[1]]), ori, laser_data, r, L)
    sim.setJointTargetVelocity(l_wheel, wl)
    sim.setJointTargetVelocity(r_wheel, wr)

    if step % 25 == 0:
        
        ax1.clear()
        ax2.clear()

        
        if hist:
            x, y = zip(*hist)
            ax1.plot(x, y, '--k', label="Trajetória")
            ax1.plot(x[0], y[0], 'go', markersize=8, label="Início")
            ax1.plot(x[-1], y[-1], 'ro', markersize=8, label="Atual")

        if laser_global:
            points_to_plot = laser_global[-N_LASER_POINTS_PLOT:]
            lx, ly = zip(*points_to_plot)
            ax1.scatter(lx, ly, s=1, c='r', alpha=0.1, label=f"Laser (últimos {N_LASER_POINTS_PLOT})")

        ax1.set_title(f"Plot de Pontos (Step {step})")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_aspect('equal')
        ax1.legend(fontsize='small')
        ax1.grid(True)

        ax1.set_xlim(MAP_ORIGIN_X, MAP_ORIGIN_X + MAP_SIZE_X)
        ax1.set_ylim(MAP_ORIGIN_Y, MAP_ORIGIN_Y + MAP_SIZE_Y)


        # Plot 2 OCCUPANCY GRID
        
        prob = 1.0 / (1.0 + np.exp(-log_odds_map)) 


        grid_vis = np.zeros_like(prob, dtype=np.uint8)
        grid_vis[prob < 0.45] = 1 
        grid_vis[prob > 0.55] = 2 
                  
        ax2.imshow(grid_vis, origin='lower', cmap=cmap_vis, extent=map_extent, vmin=0, vmax=2)

        if hist:
            xs, ys = zip(*hist)
            ax2.plot(xs, ys, color='blue', linewidth=1, label="Trajetória")
            ax2.plot(xs[0], ys[0], 'go', markersize=5, label="Início")
            ax2.plot(xs[-1], ys[-1], 'ro', markersize=5, label="Atual")

        ax2.set_title("Mapa de Ocupação (Log-Odds)")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_aspect('equal')
        ax2.legend(fontsize='small')
        
        plt.pause(0.01) 
        


# parar sim
sim.stopSimulation()
print('Program ended')

# --- MODIFICADO: LÓGICA DE PLOTAGEM FINAL ---

# Fechar a janela interativa (fig) que estava aberta
plt.close(fig) 
plt.ioff() # Garantir que estamos em modo não-interativo para salvar

print("Gerando plots finais com 600 dpi...")

# --- Plot 1: Scatter Plot (Todos os pontos) ---
fig1, ax1 = plt.subplots(figsize=(10, 10)) # Nova figura

if hist:
    x, y = zip(*hist)
    # Linha da trajetória mais fina para não poluir
    ax1.plot(x, y, '--k', label="Trajetória", linewidth=0.5)
    ax1.plot(x[0], y[0], 'go', markersize=8, label="Início")
    ax1.plot(x[-1], y[-1], 'ro', markersize=8, label="Fim")

# --- MODIFICADO: Plotar TODOS os pontos de laser ---
if len(laser_global) > 0:
    lx, ly = zip(*laser_global)
    # Usar alpha baixo e 's' pequeno para ver a densidade
    ax1.scatter(lx, ly, s=1, c='r', alpha=0.05, label="Laser (Todos os Pontos)")

ax1.legend()
ax1.set_xlabel("X [m]")
ax1.set_ylabel("Y [m]")
ax1.grid(True)
ax1.set_title("Plot de Pontos do Laser (Completo)")
ax1.set_aspect('equal')
# Garantir que os limites do mapa estão corretos
ax1.set_xlim(MAP_ORIGIN_X, MAP_ORIGIN_X + MAP_SIZE_X)
ax1.set_ylim(MAP_ORIGIN_Y, MAP_ORIGIN_Y + MAP_SIZE_Y)

# Salvar figura 1
try:
    fig1.savefig("scatter_plot_final.png", dpi=600, bbox_inches='tight')
    print("... 'scatter_plot_final.png' salvo com 600 dpi.")
except Exception as e:
    print(f"Erro ao salvar scatter plot: {e}")
plt.close(fig1) # Fechar para liberar memória


# --- Plot 2: Mapa de Ocupação Final ---
fig2, ax2 = plt.subplots(figsize=(10, 10)) # Nova figura

# Recalcular o grid visual (copiado de dentro do loop)
prob = 1.0 / (1.0 + np.exp(-log_odds_map)) 
grid_vis = np.zeros_like(prob, dtype=np.uint8) # 0 = desconhecido
grid_vis[prob < 0.45] = 1 # 1 = livre (branco)
grid_vis[prob > 0.55] = 2 # 2 = ocupado (preto)
          
ax2.imshow(grid_vis, origin='lower', cmap=cmap_vis, extent=map_extent, vmin=0, vmax=2)

if hist:
    xs, ys = zip(*hist)
    ax2.plot(xs, ys, color='blue', linewidth=0.5, label="Trajetória")
    ax2.plot(xs[0], ys[0], 'go', markersize=5, label="Início")
    ax2.plot(xs[-1], ys[-1], 'ro', markersize=5, label="Fim")

ax2.set_title("Mapa de Ocupação Final Explorado")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_aspect('equal')
ax2.legend()

# Salvar figura 2
try:
    fig2.savefig("occupancy_grid_final.png", dpi=600, bbox_inches='tight')
    print("... 'occupancy_grid_final.png' salvo com 600 dpi.")
except Exception as e:
    print(f"Erro ao salvar o mapa de ocupação: {e}")
plt.close(fig2) # Fechar para liberar memória

print("Plots finais salvos com sucesso. Script terminado.")
# O plt.show() foi removido, o script agora salva e termina.
# --- FIM DA MODIFICAÇÃO ---