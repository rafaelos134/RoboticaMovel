import numpy as np
import matplotlib.pyplot as plt
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


MAP_RESOLUTION = 0.1  # divizao das celulas
MAP_SIZE_X = 10.0     
MAP_SIZE_Y = 10.0     


MAP_ORIGIN_X = -5.0    # Canto inferior esquerdo X
MAP_ORIGIN_Y = -5.0    # Canto inferior esquerdo Y
nrows = int(MAP_SIZE_Y / MAP_RESOLUTION)
ncols = int(MAP_SIZE_X / MAP_RESOLUTION)
MAX_SCAN_DISTANCE = 5.0



LOG_ODDS_MIN = -5.0
LOG_ODDS_MAX = 5.0

map_size_m = 10.0
cell_size = 0.1                      # m por célula
ncols = int(map_size_m / cell_size)  # cols = eixo X
nrows = int(map_size_m / cell_size)  # rows = eixo Y
grid_origin = np.array([-map_size_m/2, -map_size_m/2])  # world coord do canto inferior esquerdo da célula (0,0)


world = pixel_to_world(30,34,100,10)
goal_position = np.array([world[0], world[1]])


L = 0.381  
r = 0.0975 

client = RemoteAPIClient()
sim = client.require("sim")

robotname = "kobuki"
robot = sim.getObject(f'/{robotname}')
l_wheel = sim.getObject(f'/{robotname}/wheel_left_drop_sensor/kobuki_leftMotor')
r_wheel = sim.getObject(f'/{robotname}/wheel_right_drop_sensor/kobuki_rightMotor')

# pega valores do hokuyo
hokuyo = HokuyoSensorSim.HokuyoSensorSim(sim, f"/{robotname}/fastHokuyo") 


cmap_vis = ListedColormap(['#808080', '#FFFFFF', '#000000'])
N_LASER_POINTS_PLOT = 1000
map_extent = [MAP_ORIGIN_X, MAP_ORIGIN_X + MAP_SIZE_X, 
              MAP_ORIGIN_Y, MAP_ORIGIN_Y + MAP_SIZE_Y]



hist = []            
laser_global = []    


log_odds_map = np.zeros((nrows, ncols))
last_seen = np.zeros_like(log_odds_map)


# define as cores do grid
cmap = ListedColormap(["gray", "white", "black"])




robot_path_world = [] 

LIMITE_TEMPO_SIM = 20 * 60
step = 0


contorle = controleWallFollowing.Controlador()

plt.ion()
fig, ax = plt.subplots(figsize=(6,6))

while  sim.getSimulationState() != sim.simulation_stopped:
    current_sim_time = sim.getSimulationTime()

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
    
    
    # salvar rastro do robo
    robot_path_world.append((robot_pos[0], robot_pos[1]))
    
    #controlador do robo
    wl, wr = contorle.controle_wall_following_com_escape(sim, robot, laser_data, r, L)
    
    # passa velocidade angular para roda
    sim.setJointTargetVelocity(l_wheel, wl)
    sim.setJointTargetVelocity(r_wheel, wr)

#     # # frequencia de atualizacao do mapa
    if step % 25 == 0:
    
        print("Gerando Plot 2 (Mapa de Grid)...")

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
        




     

# parar sim
sim.stopSimulation()
print('Program ended')

# # -----------------------------------------------------------------
# # Plot 1: Scatter (o que você já tinha, gera laiser.png)
# # (Este código é do seu original.py e está correto)
# # -----------------------------------------------------------------
# print("Gerando Plot 1 (Scatter)...")
# fig1 = plt.figure(figsize=(8,8), dpi=100)
# ax1 = fig1.add_subplot(111, aspect='equal')

# # 'hist' é a variável do seu script original.py
# x, y = zip(*hist)
# ax1.plot(x, y, '--k', label="Trajetória")
# ax1.plot(x[0], y[0], 'go', markersize=10, label="Início")
# ax1.plot(x[-1], y[-1], 'ro', markersize=10, label="Fim")

# # 'laser_global' é a variável do seu script original.py
# if len(laser_global) > 0:
#     lx, ly = zip(*laser_global)
#     ax1.scatter(lx, ly, s=2, c='r', alpha=0.3, label="Laser")

# ax1.legend()
# ax1.set_xlabel("X [m]")
# ax1.set_ylabel("Y [m]")
# ax1.grid(True)
# ax1.set_title("Plot de Pontos do Laser (Scatter)")
# plt.show()


# # -----------------------------------------------------------------
# # Plot 2: Mapa de Grade (gera grid.png)
# # (Este é o código corrigido e integrado)
# # -----------------------------------------------------------------
# print("Gerando Plot 2 (Mapa de Grade)...")

# # Converter log-odds para probabilidade (sua fórmula está correta)
# # 'log_odds_map' foi criada e atualizada no seu loop 'while'
# prob = 1.0 / (1.0 + np.exp(-log_odds_map)) 

# # Criar mapa visual (0=desconhecido, 1=livre, 2=ocupado)
# grid_vis = np.zeros_like(prob, dtype=np.uint8) # 0 = desconhecido
# grid_vis[prob < 0.45] = 1 # 1 = livre (branco)
# grid_vis[prob > 0.55] = 2 # 2 = ocupado (preto)

# # Definir um colormap: 0=cinza, 1=branco, 2=preto
# # (Substitui a variável 'cmap' indefinida)
# cmap = ListedColormap(['#808080', '#FFFFFF', '#000000'])

# plt.ioff()
# fig2, ax2 = plt.subplots(figsize=(7,7))

# extent = [MAP_ORIGIN_X, MAP_ORIGIN_X + MAP_SIZE_X, 
#           MAP_ORIGIN_Y, MAP_ORIGIN_Y + MAP_SIZE_Y]
          
# ax2.imshow(grid_vis, origin='lower', cmap=cmap, extent=extent, vmin=0, vmax=2)


# if hist:
#     xs, ys = zip(*hist)
#     ax2.plot(xs, ys, color='blue', linewidth=1, label="Trajetória")
#     ax2.plot(xs[-1], ys[-1], 'ro', markersize=5, label="Fim") # Ponto final


# ax2.set_title("Mapa Final Explorado")
# ax2.set_xlabel("X (m)")
# ax2.set_ylabel("Y (m)")
# ax2.set_aspect('equal')
# ax2.legend()
# plt.show()
