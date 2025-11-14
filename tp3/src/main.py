import numpy as np
import matplotlib     
matplotlib.use("Qt5Agg")  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import networkx as nx
import numpy as np

from matplotlib.colors import ListedColormap

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


 
# funcoes em outros arquivos
import HokuyoSensorSim
import controleWallFollowing
import logOdds



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


contorle = controleWallFollowing.Controlador()

while sim.getSimulationState() != sim.simulation_stopped:

    sim.step()
    step +=1

    if step > 1000:
        print(f"Numero de passos maximo atingido. Parando a simulação.")
        break

    pos = sim.getObjectPosition(robot, sim.handle_world)
    ori = sim.getObjectOrientation(robot, sim.handle_world)
    
    
    hist.append([pos[0], pos[1]])
    
    laser_data = hokuyo.getSensorData()
    laser_global.extend(HokuyoSensorSim.transform_laser_to_global(laser_data, pos, ori))

    robot_pos = [pos[0], pos[1], ori[2]]
    
    
    laser_data_swapped = [[d, a] for a, d in laser_data]
        
    log_odds_map, last_seen = logOdds.update_map_log_odds(
        log_odds_map,
        robot_pos,
        laser_data_swapped,
        step,
        last_seen)
    
    
    # salvar rastro em coords do mundo
    robot_path_world.append((robot_pos[0], robot_pos[1]))
    wl, wr = contorle.controle_wall_following_com_escape(sim, robot, laser_data, r, L)
    
    # wl, wr =  controle_campos(world, goal_position, np.array([pos[0], pos[1]]), ori, laser_data, r, L)
    sim.setJointTargetVelocity(l_wheel, wl)
    sim.setJointTargetVelocity(r_wheel, wr)

    if step % 25 == 0: # Frequência de atualização aumentada
        
        # --- Limpa os eixos para redesenhar ---
        ax1.clear()
        ax2.clear()

        # --- PLOT 1: Scatter (Pontos do Laser e Trajetória) ---
        if hist:
            x, y = zip(*hist)
            ax1.plot(x, y, '--k', label="Trajetória")
            ax1.plot(x[0], y[0], 'go', markersize=8, label="Início")
            ax1.plot(x[-1], y[-1], 'ro', markersize=8, label="Atual")

        if laser_global:
            # Plota apenas os últimos N pontos para performance
            points_to_plot = laser_global[-N_LASER_POINTS_PLOT:]
            lx, ly = zip(*points_to_plot)
            ax1.scatter(lx, ly, s=1, c='r', alpha=0.1, label=f"Laser (últimos {N_LASER_POINTS_PLOT})")

        ax1.set_title(f"Plot de Pontos (Step {step})")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_aspect('equal')
        ax1.legend(fontsize='small')
        ax1.grid(True)
        # Define limites fixos para o scatter plot não "pular"
        ax1.set_xlim(MAP_ORIGIN_X, MAP_ORIGIN_X + MAP_SIZE_X)
        ax1.set_ylim(MAP_ORIGIN_Y, MAP_ORIGIN_Y + MAP_SIZE_Y)


        # --- PLOT 2: Mapa de Ocupação (Grid Map) ---
        
        # Converter log-odds para probabilidade
        prob = 1.0 / (1.0 + np.exp(-log_odds_map)) 

        # Criar mapa visual (0=desconhecido, 1=livre, 2=ocupado)
        grid_vis = np.zeros_like(prob, dtype=np.uint8) # 0 = desconhecido
        grid_vis[prob < 0.45] = 1 # 1 = livre (branco)
        grid_vis[prob > 0.55] = 2 # 2 = ocupado (preto)
                  
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
        
        # --- ATUALIZA A JANELA ---
        plt.pause(0.01) # Pausa rápida para a GUI processar eventos
        # (Opcional, mas ajuda em alguns backends)
        # fig.canvas.draw()
        # fig.canvas.flush_events()
        


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
    fig2.savefig("images/occupancy_grid_final.png", dpi=600, bbox_inches='tight')
except Exception as e:
    print(f"Erro ao salvar o mapa de ocupação: {e}")
plt.close(fig2) # Fechar para liberar memória

print("Plots finais salvos com sucesso. Script terminado.")