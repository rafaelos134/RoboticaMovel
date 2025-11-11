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



# codigo de aluno no moodle

class HokuyoSensorSim(object): 
    _sim = None
    
    _base_name = "" 

    _angles_lua = np.empty(0) 
    _is_initialized_angles = False 
    ANGLE_SIGNAL = 'signal.hokuyo_angle_data' 
    
    _base_obj = None 
    _is_range_data = True
    _vision_sensors_obj = []

    def __init__(self, sim, base_name, is_range_data=True):
        self._sim = sim
        self._base_name = base_name
        self._is_range_data = is_range_data

        if "fastHokuyo" not in base_name:
            raise ValueError(
                f"ERR: fastHokuyo must be in the base object name. Ex: `/kobuki/fastHokuyo`"
            )

        self._base_obj = sim.getObject(base_name)
        if self._base_obj == -1:
            raise ValueError(
                f"ERR: base_obj ({self._base_obj}) is not a valid name in the simulation"
            )

        self._vision_sensors_obj = [
            sim.getObject(f'{base_name}/fastHokuyo_sensor1'),
            sim.getObject(f'{base_name}/fastHokuyo_sensor2'),
        ]

        if any(obj == -1 for obj in self._vision_sensors_obj):
            raise ValueError(
                f"ERR: the _vision_sensors_obj names are not valid in the simulation"
            )

    def get_is_range_data(self) -> bool:
        return self._is_range_data

    def set_is_range_data(self, is_range_data: bool) -> None:
        self._is_range_data = is_range_data

    def _initialize_angles_from_lua(self):

        # Essas variáveis agora são utilizadas apenas quando os valores do laser 
        # não são desempacotados diretamente do sensor, como um fallback em caso de falha
        angle_min = -120 * math.pi / 180
        angle_increment = (240 / 684) * math.pi / 180
        total_steps = 684
        
        for i in range(15):
            try:
                if self._vision_sensors_obj and self._vision_sensors_obj[0] != -1:
                    self._sim.readVisionSensor(self._vision_sensors_obj[0]) 

                angles_packed = self._sim.getBufferProperty(self._sim.handle_scene, self.ANGLE_SIGNAL, {'noError' : True})
                
                if angles_packed:
                    self._angles_lua = np.array(self._sim.unpackFloatTable(angles_packed))
                    self._is_initialized_angles = True
                    print(f"Precise sensor angles read on attempt {i+1} ({self._angles_lua.size} laser beam readings)")
                    return True
                
                time.sleep(0.01)

            except Exception as e:
                pass 
                
        self._angles_lua = np.arange(angle_min, angle_min + total_steps * angle_increment, angle_increment)
        self._is_initialized_angles = True
        print("ALERT: Communication via buffer property failed. Using approximation. Map may be blurred.")
        return False

    def getSensorData(self):
        
        if not self._is_initialized_angles:
            self._initialize_angles_from_lua()
        
        sensor_data = []
        angle_idx = 0
        angles_to_use = self._angles_lua
        
        if angles_to_use.size == 0:
            return np.empty((0, 2))
        
        for vision_sensor in self._vision_sensors_obj:
            
            result = self._sim.readVisionSensor(vision_sensor)
            if not isinstance(result, (list, tuple)) or len(result) != 3: continue 
                
            r, t, u = result
            if u:
                for j in range(int(u[1])): 
                    for k in range(int(u[0])): 
                        w_idx = 2 + 4 * (j * int(u[0]) + k)
                        v_dist = u[w_idx + 3] 
                        if angle_idx < angles_to_use.size:
                            current_angle = angles_to_use[angle_idx]
                            sensor_data.append([current_angle, v_dist])
                            angle_idx += 1
                        else:
                            break 
                    if angle_idx >= angles_to_use.size: break

        return np.array(sensor_data) if sensor_data else np.empty((0, 2))
    
def transform_laser_to_global(laser_data, robot_pos, robot_ori):

    x_r, y_r = robot_pos[0], robot_pos[1]
    theta_r = robot_ori[2]  # orientação em torno de z

    global_points = []

    

    for ang, dist in laser_data:
        if dist > 0.01 and dist < 5:
            # coordenadas no robô
            x_local = dist * np.cos(ang)
            y_local = dist * np.sin(ang)

            # transformação para global
            x_global = x_r + x_local*np.cos(theta_r) - y_local*np.sin(theta_r)
            y_global = y_r + x_local*np.sin(theta_r) + y_local*np.cos(theta_r)

            global_points.append([x_global, y_global])

    return global_points





# verificar funcao
def pixel_to_world(x_px, y_px, img_size=64, world_size=10):
        scale = world_size / img_size
        x_world = ((x_px - img_size/2) * scale)
        y_world = (-(y_px - img_size/2) * scale) 
        return x_world, y_world


def att_force(q, goal, katt=.01):
    return katt*(goal - q)

def rep_force(q, laser_data, R=3, krep=.1):
    # Se lista estiver vazia
    if not laser_data:
        return np.zeros(2)

    laser_data = np.array(laser_data, dtype=float)

    # Garante que tenha 3 colunas (x, y, r)
    if laser_data.shape[1] == 2:
        # adiciona uma coluna de zeros (raio)
        laser_data = np.hstack([laser_data, np.zeros((laser_data.shape[0], 1))])

    # Vetor posição relativa robô -> obstáculo
    v = q - laser_data[:, :2]

    # Distância ao obstáculo (menos o raio)
    d = np.linalg.norm(v, axis=1) - laser_data[:, 2]
    d = np.maximum(d, 1e-6)  # evita divisão por zero
    d = d.reshape(-1, 1)

    # Força repulsiva
    rep = (1/d**2) * ((1/d) - (1/R)) * (v/d)

    # Zera onde fora do raio de influência
    rep[d.flatten() > R, :] = 0.0

    # Retorna soma das forças
    return krep * np.sum(rep, axis=0)


def controle(laser_data,r,L):
     # Controle simples de desvio
        v, w = 0, 0
        frente = int(len(laser_data) / 2)
        lado_direito = int(len(laser_data) * 1 / 4)
        lado_esquerdo = int(len(laser_data) * 3 / 4)

        if laser_data[frente, 1] > 2:
            v = .5
            w = 0
        elif laser_data[lado_direito, 1] > 1:
            v = 0
            w = np.deg2rad(-45)
        elif laser_data[lado_esquerdo, 1] > 1:
            v = 0
            w = np.deg2rad(45)

        # Modelo cinemático
        wl = v / r - (w * L) / (2 * r)
        wr = v / r + (w * L) / (2 * r)
        
        return wl,wr



def controle_campos(word, goal_position, robot_pos, robot_ori, laser_data, r, L):
    # constantes -> verificar
    K_att = 50
    REP_RADIUS = 0.6
    REP_GAIN = 15

    MAX_LINEAR_VEL = 0.8
    MAX_ANGULAR_VEL = np.deg2rad(90)

    LINEAR_GAIN = 0.25
    ANGULAR_GAIN = 2.0
    GOAL_TOL = 0.20

    laser_global = transform_laser_to_global(laser_data, robot_pos, robot_ori)
    f_att = att_force(robot_pos, goal_position,K_att)
    f_rep = rep_force(robot_pos, laser_global, R = REP_RADIUS, krep = REP_GAIN)

    f_total = f_att + f_rep

    # CONTROLADOR DESAI ET AL. (1998) -> verificar se realmente está correto
    xd, yd = f_total
    theta = robot_ori[2]
    d = L / 2.0  

    J = np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta) / d, np.cos(theta) / d]
    ])

    v_cmd, w_cmd = J @ np.array([xd, yd])

    v_cmd *= LINEAR_GAIN
    w_cmd *= ANGULAR_GAIN

    theta_d = np.arctan2(f_total[1], f_total[0])
    erro_theta = np.arctan2(np.sin(theta_d - theta), np.cos(theta_d - theta))
    if abs(erro_theta) > np.deg2rad(45):
        v_cmd *= 0.25

    v_cmd = np.clip(v_cmd, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
    w_cmd = np.clip(w_cmd, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

    # prev_v_cmd = v_cmd

    # cinematica
    v_r = (2.0 * v_cmd + w_cmd * L) / (2.0 * r)
    v_l = (2.0 * v_cmd - w_cmd * L) / (2.0 * r)

    return v_l, v_r

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
        # log_odds_map[posi_yGrid, posi_xGrid] += l_occ

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


# verifica se o robo chegou no objetivo e finaliza o loop
    # if np.linalg.norm(goal_position - robot_pos) <= GOAL_TOL:
    #     print("Objetivo alcancado.")
    #     break



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

for step in range(1000):
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

    # log_odds_map = update_map_log_odds(log_odds_map, robot_pos, laser_data)

    # salvar rastro em coords do mundo
    robot_path_world.append((robot_pos[0], robot_pos[1]))
    
    wl, wr =  controle_campos(world, goal_position, np.array([pos[0], pos[1]]), ori, laser_data, r, L)
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


