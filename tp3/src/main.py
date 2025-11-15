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
    
    
    # salvar rastro do robo
    robot_path_world.append((robot_pos[0], robot_pos[1]))
    
    #controlador do robo
    wl, wr = contorle.controle_wall_following_com_escape(sim, robot, laser_data, r, L)
    
    # passa velocidade angular para roda
    sim.setJointTargetVelocity(l_wheel, wl)
    sim.setJointTargetVelocity(r_wheel, wr)

    # frequencia de atualizacao do mapa
    if step % 25 == 0:
    
        ax1.clear()
        ax2.clear()

        # --- PLOT 1: Scatter e trajetoria ---
        if hist and len(hist) > 1:
            x, y = zip(*hist)
            ax1.plot(x, y, '--k', label="Trajetória")
            ax1.plot(x[0], y[0], 'go', markersize=8, label="Início")
            ax1.plot(x[-1], y[-1], 'ro', markersize=8, label="Atual")

        if laser_global and len(laser_global) > 0:
            points_to_plot = laser_global[-N_LASER_POINTS_PLOT:]
            lx, ly = zip(*points_to_plot)
            ax1.scatter(lx, ly, s=1, c='r', alpha=0.1,
                        label=f"Laser (últimos {N_LASER_POINTS_PLOT})")

        ax1.set_title(f"Plot de Pontos (Step {step})")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_aspect('equal')
        ax1.legend(fontsize='small')
        ax1.grid(True)
        ax1.set_xlim(MAP_ORIGIN_X, MAP_ORIGIN_X + MAP_SIZE_X)
        ax1.set_ylim(MAP_ORIGIN_Y, MAP_ORIGIN_Y + MAP_SIZE_Y)

        # --- PLOT 2: Grid Map ---
        prob = 1.0 / (1.0 + np.exp(-log_odds_map))
        grid_vis = np.zeros_like(prob, dtype=np.uint8)
        grid_vis[prob < 0.45] = 1 # Célula livre (branco)
        grid_vis[prob > 0.55] = 2 # Célula ocupada (preto)
        # Células entre 0.45 e 0.55 permanecem 0 (cinza/desconhecido)

        ax2.imshow(grid_vis, origin='lower', cmap=cmap_vis,
                extent=map_extent, vmin=0, vmax=2)

        if hist and len(hist) > 1:
            xs, ys = zip(*hist)
            ax2.plot(xs, ys, color='blue', linewidth=1, label="Trajetória")
            ax2.plot(xs[0], ys[0], 'go', markersize=5, label="Início")
            ax2.plot(xs[-1], ys[-1], 'ro', markersize=5, label="Atual")

        ax2.set_title("Mapa de Ocupação (Log-Odds)")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_aspect('equal')
        ax2.legend(fontsize='small')
        
        # --- CORREÇÃO PRINCIPAL ---
        
        # 1. ATUALIZE A TELA (ISTO É O CORRETO)
        plt.pause(0.01)
                


# parar sim
sim.stopSimulation()
print('Program ended')

# --- SALVAR AS FIGURAS FINAIS AQUI ---
print("Salvando imagens finais...")
try:
    fig1.savefig("scatter_plot_final.png", dpi=600, bbox_inches='tight')
    # Certifique-se que a pasta "images" exista!
    fig2.savefig("images/occupancy_grid_final.png", dpi=600, bbox_inches='tight')
    print("Imagens salvas com sucesso.")
except FileNotFoundError:
    print("Erro: A pasta 'images/' não foi encontrada. Verifique o caminho.")
    
