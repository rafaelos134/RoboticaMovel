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
    
    
# codigo do moodle Iago

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
    
   
def controle_wall_following_com_escape(sim, robot, laser_data, r, L):
    """
    Controlador 'Right Wall Following' com lógica de escape quando o robô
    fica preso.

    Esta função agora é 'stateful' (armazena seu próprio estado) para
    gerenciar a detecção de 'preso' e a máquina de estados de escape.
    """
    
    # Referência à própria função para armazenar o estado
    f = controle_wall_following_com_escape

    # --- Inicialização do Estado (executado apenas na primeira chamada) ---
    if not hasattr(f, "state"):
        print("Controlador com escape: Inicializando estado...")
        f.state = "FOLLOWING"           # Estado atual: "FOLLOWING", "ESCAPE_TURN", "ESCAPE_MOVE"
        f.last_pos = None               # Última posição [x, y, z] para checar se está preso
        f.last_pos_time = 0.0           # Tempo da última checagem de posição
        f.stuck_timer = 0.0             # Acumulador de tempo que o robô está potencialmente preso
        
        f.target_yaw = None             # Ângulo-alvo para o giro de 90 graus
        f.escape_move_endtime = 0.0     # Tempo final para o movimento à frente
        
        # Memória do seguidor de parede (do seu código original)
        f.ultima_direita = 0.5          

    # --- Constantes do Sistema de Escape ---
    STUCK_CHECK_INTERVAL = 1.0  # (s) A cada quanto tempo checamos o movimento
    STUCK_TIME_LIMIT = 5.0      # (s) Tempo parado para ser considerado "preso"
    STUCK_MOVE_THRESHOLD = 0.1  # (m) Movimento mínimo para não ser "preso"
    ESCAPE_TURN_ANGLE = np.deg2rad(90) # (rad) Girar 90 graus à esquerda
    ESCAPE_MOVE_TIME = 1.5      # (s) Andar à frente por este tempo
    ESCAPE_MOVE_VEL = 0.3       # (m/s) Velocidade linear durante o escape
    ESCAPE_TURN_VEL = np.deg2rad(45) # (rad/s) Velocidade angular durante o giro

    # --- Constantes do Wall Following (do seu código) ---
    DISTANCIA_DESEJADA = 0.4
    KP_ANGULAR = 2
    VEL_LINEAR_FIXA = 1.0
    MAX_LINEAR_VEL = 1.0
    MAX_ANGULAR_VEL = np.deg2rad(90)
    LIMITE_FRONTAL = 0.6
    ZONA_MORTA_ERRO = 0.05
    DISTANCIA_SEM_PAREDE = 2.0

    # --- Obter dados atuais do robô (via 'sim') ---
    current_time = sim.getSimulationTime()
    current_pos = sim.getObjectPosition(robot, -1) # Posição [x, y, z]
    current_orient_euler = sim.getObjectOrientation(robot, -1) # Euler [a, b, g]
    current_yaw = current_orient_euler[2] # Yaw (orientação em Z)

    # --- Lógica de Detecção de "Preso" (só checa se estiver em "FOLLOWING") ---
    if f.state == "FOLLOWING":
        if f.last_pos is None:
            # Primeira execução, apenas armazena os dados
            f.last_pos = current_pos
            f.last_pos_time = current_time
        
        # Verifica se passou o tempo de checagem
        time_since_last_check = current_time - f.last_pos_time
        if time_since_last_check > STUCK_CHECK_INTERVAL:
            dist_moved = np.linalg.norm(np.array(current_pos) - np.array(f.last_pos))
            
            if dist_moved < STUCK_MOVE_THRESHOLD:
                # Não moveu o suficiente, incrementa o tempo "preso"
                f.stuck_timer += time_since_last_check
                # print(f"Stuck timer: {f.stuck_timer:.1f}s") # Debug
            else:
                # Moveu, reseta o contador
                f.stuck_timer = 0.0
            
            # Atualiza a última posição e tempo
            f.last_pos = current_pos
            f.last_pos_time = current_time

        # --- GATILHO DO ESCAPE ---
        if f.stuck_timer > STUCK_TIME_LIMIT:
            print(f"Robô preso por {f.stuck_timer:.1f}s! Iniciando manobra de escape.")
            f.state = "ESCAPE_TURN"
            f.stuck_timer = 0.0 # Reseta o contador
            
            # Calcula o yaw alvo (90 graus à esquerda do yaw atual)
            target_yaw_unnormalized = current_yaw + ESCAPE_TURN_ANGLE
            # Normaliza o ângulo para ficar entre -pi e +pi
            f.target_yaw = np.arctan2(np.sin(target_yaw_unnormalized), np.cos(target_yaw_unnormalized))
            print(f"  Yaw atual: {np.rad2deg(current_yaw):.1f} deg, Alvo: {np.rad2deg(f.target_yaw):.1f} deg")


    # --- Máquina de Estados de Controle ---

    if f.state == "ESCAPE_TURN":
        # --- Estado 1: Girar 90 graus à esquerda ---
        
        # Calcula o erro de orientação (menor caminho)
        erro_yaw = f.target_yaw - current_yaw
        erro_yaw = np.arctan2(np.sin(erro_yaw), np.cos(erro_yaw))
        
        if abs(erro_yaw) < np.deg2rad(5):
            # Chegou perto do ângulo-alvo
            print("  Giro de escape completo.")
            f.state = "ESCAPE_MOVE"
            f.escape_move_endtime = current_time + ESCAPE_MOVE_TIME # Seta o timer
            v_cmd = 0.0
            w_cmd = 0.0
        else:
            # Gira no lugar para a esquerda (sinal positivo)
            v_cmd = 0.0
            w_cmd = ESCAPE_TURN_VEL # Velocidade angular fixa
            
    elif f.state == "ESCAPE_MOVE":
        # --- Estado 2: Mover para frente por um tempo ---
        
        if current_time < f.escape_move_endtime:
            # Anda para frente
            v_cmd = ESCAPE_MOVE_VEL
            w_cmd = 0.0
        else:
            # Tempo de escape acabou
            print("  Movimento de escape completo. Retornando ao Wall Following.")
            f.state = "FOLLOWING"
            f.last_pos = None # Força a re-inicialização do detector de "preso"
            v_cmd = 0.0
            w_cmd = 0.0
            
    elif f.state == "FOLLOWING":
        # --- Estado 0: Lógica Original (Wall Following) ---
        
        # Processamento do laser (copiado do seu código)
        if isinstance(laser_data, dict) and "ranges" in laser_data:
            ranges = np.array(laser_data["ranges"], dtype=float)
        else:
            ranges = np.array(laser_data[:, 1], dtype=float)

        ranges = np.nan_to_num(ranges, posinf=100.0, neginf=0.0)
        num = len(ranges)
        
        if num == 0:
            v_cmd, w_cmd = 0.0, 0.0
        else:
            c = num // 2
            fatia_frente = int(num * 0.08)
            frente_ranges = ranges[c - fatia_frente : c + fatia_frente]
            dist_frente = np.min(frente_ranges) if frente_ranges.size > 0 else 100.0

            fatia_direita = int(num * 0.10)
            direita_ranges = ranges[0 : fatia_direita]
            validas = direita_ranges[direita_ranges < DISTANCIA_SEM_PAREDE]
            if len(validas) > 0:
                dist_direita = np.min(validas) # Simplesmente pega o ponto mais próximo
            else:
                dist_direita = DISTANCIA_SEM_PAREDE

            # --- Memória curta (MODIFICADO para usar f.ultima_direita) ---
            if dist_direita > f.ultima_direita * 1.5:
                dist_direita = f.ultima_direita * 1.1
            f.ultima_direita = dist_direita # Atualiza o estado persistente

            if dist_frente < LIMITE_FRONTAL:
                # --- NOVO: Controle Proporcional de Desvio ---
                
                # Fator de 0.0 (muito perto) a 1.0 (no limite)
                fator_proximidade = dist_frente / LIMITE_FRONTAL
                
                # Velocidade linear é proporcional à distância (mais perto, mais devagar)
                v_cmd = VEL_LINEAR_FIXA * fator_proximidade
                v_cmd = max(v_cmd, 0.05) # Garante uma velocidade mínima
                
                # Velocidade angular é INVERSAMENTE proporcional (mais perto, vira mais)
                # Sempre vira para a esquerda (positivo) para contornar
                fator_virada = 1.0 - fator_proximidade
                w_cmd = MAX_ANGULAR_VEL * fator_virada
                
                # (Opcional, mas recomendado) Se a parede da direita sumir, vira mais suave
                if dist_direita > DISTANCIA_SEM_PAREDE:
                     w_cmd *= 0.5 # Reduz a virada se a direita estiver livre
                
                
                # --- FIM DO NOVO BLOCO ---
            elif dist_direita > DISTANCIA_SEM_PAREDE:
                v_cmd = 0.4
                w_cmd = 0.0
            else:
                erro = DISTANCIA_DESEJADA - dist_direita
                if abs(erro) < ZONA_MORTA_ERRO:
                    erro = 0.0
                w_cmd = KP_ANGULAR * erro
                fator_reducao = 1.0 - (abs(w_cmd) / MAX_ANGULAR_VEL)
                v_cmd = VEL_LINEAR_FIXA * fator_reducao
                v_cmd = max(v_cmd, 0.1)

    # --- Saturação Final (para todos os estados) ---
    v_cmd = np.clip(v_cmd, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
    w_cmd = np.clip(w_cmd, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

    # --- Cinemática diferencial (para todos os estados) ---
    v_r = (2.0 * v_cmd + w_cmd * L) / (2.0 * r)
    v_l = (2.0 * v_cmd - w_cmd * L) / (2.0 * r)

    return v_l, v_r

#melhor desparado
def controle_wall_following_com_escape(sim, robot, laser_data, r, L):
    """
    Controlador 'Right Wall Following' com lógica de escape melhorada (com ré)
    e controle frontal proporcional para ambientes complexos.
    
    Esta função agora é 'stateful' (armazena seu próprio estado) para
    gerenciar a detecção de 'preso' e a máquina de estados de escape.
    """
    
    # Referência à própria função para armazenar o estado
    f = controle_wall_following_com_escape

    # --- Inicialização do Estado (executado apenas na primeira chamada) ---
    if not hasattr(f, "state"):
        print("Controlador com escape: Inicializando estado...")
        # --- MODIFICADO --- Estados: "FOLLOWING", "ESCAPE_REVERSE", "ESCAPE_TURN", "ESCAPE_MOVE"
        f.state = "FOLLOWING"
        f.last_pos = None
        f.last_pos_time = 0.0
        f.stuck_timer = 0.0
        
        f.target_yaw = None
        f.escape_move_endtime = 0.0
        f.escape_reverse_endtime = 0.0 # --- NOVO --- Timer para a marcha à ré
        
        f.ultima_direita = 0.5

    # --- Constantes do Sistema de Escape ---
    STUCK_CHECK_INTERVAL = 1.0  # (s) A cada quanto tempo checamos o movimento
    STUCK_TIME_LIMIT = 5.0      # (s) Tempo parado para ser considerado "preso"
    STUCK_MOVE_THRESHOLD = 0.1  # (m) Movimento mínimo para não ser "preso"
    
    # --- NOVO: Lógica de Ré ---
    ESCAPE_REVERSE_TIME = 1.5   # (s) Tempo de marcha à ré
    ESCAPE_REVERSE_VEL = -0.2   # (m/s) Velocidade negativa (ré)
    
    ESCAPE_TURN_ANGLE = np.deg2rad(90) # (rad) Girar 90 graus à esquerda
    ESCAPE_TURN_VEL = np.deg2rad(45) # (rad/s) Velocidade angular durante o giro
    
    # --- MODIFICADO --- Aumentei o tempo de movimento para frente
    ESCAPE_MOVE_TIME = 1.5      # (s) Andar à frente por este tempo
    ESCAPE_MOVE_VEL = 0.3       # (m/s) Velocidade linear durante o escape


    # --- Constantes do Wall Following (Ajustadas) ---
    # --- MODIFICADO --- Distância menor para "grudar" mais na parede
    DISTANCIA_DESEJADA = 0.4
    KP_ANGULAR = 2
    VEL_LINEAR_FIXA = 1.0
    MAX_LINEAR_VEL = 1.0
    MAX_ANGULAR_VEL = np.deg2rad(90)
    # --- MODIFICADO --- Limite frontal menor para reagir mais perto
    LIMITE_FRONTAL = 0.6
    ZONA_MORTA_ERRO = 0.05
    DISTANCIA_SEM_PAREDE = 2.0

    # --- Obter dados atuais do robô (via 'sim') ---
    current_time = sim.getSimulationTime()
    current_pos = sim.getObjectPosition(robot, -1) # Posição [x, y, z]
    current_orient_euler = sim.getObjectOrientation(robot, -1) # Euler [a, b, g]
    current_yaw = current_orient_euler[2] # Yaw (orientação em Z)

    # --- Lógica de Detecção de "Preso" (só checa se estiver em "FOLLOWING") ---
    if f.state == "FOLLOWING":
        if f.last_pos is None:
            f.last_pos = current_pos
            f.last_pos_time = current_time
        
        time_since_last_check = current_time - f.last_pos_time
        if time_since_last_check > STUCK_CHECK_INTERVAL:
            dist_moved = np.linalg.norm(np.array(current_pos) - np.array(f.last_pos))
            
            if dist_moved < STUCK_MOVE_THRESHOLD:
                f.stuck_timer += time_since_last_check
            else:
                f.stuck_timer = 0.0
            
            f.last_pos = current_pos
            f.last_pos_time = current_time

        # --- GATILHO DO ESCAPE ---
        if f.stuck_timer > STUCK_TIME_LIMIT:
            print(f"Robô preso por {f.stuck_timer:.1f}s! Iniciando manobra de escape.")
            # --- MODIFICADO --- Inicia dando ré, não virando
            f.state = "ESCAPE_REVERSE"
            f.escape_reverse_endtime = current_time + ESCAPE_REVERSE_TIME
            f.stuck_timer = 0.0 # Reseta o contador
            

    # --- Máquina de Estados de Controle ---

    # --- NOVO --- Estado 1: Dar ré para criar espaço
    if f.state == "ESCAPE_REVERSE":
        if current_time < f.escape_reverse_endtime:
            # Dando ré
            v_cmd = ESCAPE_REVERSE_VEL
            w_cmd = 0.0
        else:
            # Ré completa, prepara para virar
            print("  Ré de escape completa. Iniciando giro.")
            f.state = "ESCAPE_TURN"
            
            # --- MODIFICADO --- Cálculo do yaw alvo movido para cá
            target_yaw_unnormalized = current_yaw + ESCAPE_TURN_ANGLE
            f.target_yaw = np.arctan2(np.sin(target_yaw_unnormalized), np.cos(target_yaw_unnormalized))
            print(f"  Yaw atual: {np.rad2deg(current_yaw):.1f} deg, Alvo: {np.rad2deg(f.target_yaw):.1f} deg")
            
            v_cmd = 0.0 # Para neste frame
            w_cmd = 0.0
            
    elif f.state == "ESCAPE_TURN":
        # --- Estado 2: Girar 90 graus à esquerda ---
        
        erro_yaw = f.target_yaw - current_yaw
        erro_yaw = np.arctan2(np.sin(erro_yaw), np.cos(erro_yaw))
        
        if abs(erro_yaw) < np.deg2rad(5):
            print("  Giro de escape completo.")
            f.state = "ESCAPE_MOVE"
            f.escape_move_endtime = current_time + ESCAPE_MOVE_TIME
            v_cmd = 0.0
            w_cmd = 0.0
        else:
            v_cmd = 0.0
            w_cmd = ESCAPE_TURN_VEL
            
    elif f.state == "ESCAPE_MOVE":
        # --- Estado 3: Mover para frente por um tempo ---
        
        if current_time < f.escape_move_endtime:
            v_cmd = ESCAPE_MOVE_VEL
            w_cmd = 0.0
        else:
            print("  Movimento de escape completo. Retornando ao Wall Following.")
            f.state = "FOLLOWING"
            f.last_pos = None
            v_cmd = 0.0
            w_cmd = 0.0
            
    elif f.state == "FOLLOWING":
        # --- Estado 0: Lógica Original (Wall Following) ---
        
        if isinstance(laser_data, dict) and "ranges" in laser_data:
            ranges = np.array(laser_data["ranges"], dtype=float)
        else:
            ranges = np.array(laser_data[:, 1], dtype=float)

        ranges = np.nan_to_num(ranges, posinf=100.0, neginf=0.0)
        num = len(ranges)
        
        if num == 0:
            v_cmd, w_cmd = 0.0, 0.0
        else:
            c = num // 2
            fatia_frente = int(num * 0.08)
            frente_ranges = ranges[c - fatia_frente : c + fatia_frente]
            dist_frente = np.min(frente_ranges) if frente_ranges.size > 0 else 100.0

            fatia_direita = int(num * 0.10)
            direita_ranges = ranges[0 : fatia_direita]
            validas = direita_ranges[direita_ranges < DISTANCIA_SEM_PAREDE]
            
            if len(validas) > 0:
                # --- MODIFICADO --- Simplificado para min(), mais robusto a "pernas"
                dist_direita = np.min(validas)
            else:
                dist_direita = DISTANCIA_SEM_PAREDE

            if dist_direita > f.ultima_direita * 1.5:
                dist_direita = f.ultima_direita * 1.1
            f.ultima_direita = dist_direita

            # --- Controle principal (copiado do seu código) ---
            
            # --- MODIFICADO: Lógica de desvio frontal ---
            if dist_frente < LIMITE_FRONTAL:
                # Controle Proporcional de Desvio:
                # Quanto mais perto (fator_proximidade -> 0), 
                # mais devagar (v_cmd -> 0) e mais rápido vira (w_cmd -> MAX)
                
                fator_proximidade = dist_frente / LIMITE_FRONTAL # (de 0.0 a 1.0)
                
                v_cmd = VEL_LINEAR_FIXA * fator_proximidade
                v_cmd = max(v_cmd, 0.05) # "Creep" (andar devagar)
                
                fator_virada = 1.0 - fator_proximidade # (de 0.0 a 1.0)
                w_cmd = MAX_ANGULAR_VEL * fator_virada # Vira à esquerda
            # --- FIM DA MODIFICAÇÃO ---
            
            elif dist_direita > DISTANCIA_SEM_PAREDE:
                v_cmd = 0.4
                w_cmd = 0.0 # Segue reto se perder a parede
            else:
                # Controle P para seguir a parede
                erro = DISTANCIA_DESEJADA - dist_direita
                if abs(erro) < ZONA_MORTA_ERRO:
                    erro = 0.0
                w_cmd = KP_ANGULAR * erro
                fator_reducao = 1.0 - (abs(w_cmd) / MAX_ANGULAR_VEL)
                v_cmd = VEL_LINEAR_FIXA * fator_reducao
                v_cmd = max(v_cmd, 0.1)

    # --- Saturação Final (para todos os estados) ---
    v_cmd = np.clip(v_cmd, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
    w_cmd = np.clip(w_cmd, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

    # --- Cinemática diferencial (para todos os estados) ---
    v_r = (2.0 * v_cmd + w_cmd * L) / (2.0 * r)
    v_l = (2.0 * v_cmd - w_cmd * L) / (2.0 * r)

    return v_l, v_r


def controle_wall_following_com_escape(sim, robot, laser_data, r, L):
    """
    Controlador 'Right Wall Following' com lógica de escape melhorada (com ré)
    e controle frontal proporcional para ambientes complexos.
    
    Esta função agora é 'stateful' (armazena seu próprio estado) para
    gerenciar a detecção de 'preso' e a máquina de estados de escape.
    """
    
    # Referência à própria função para armazenar o estado
    f = controle_wall_following_com_escape

    # --- Inicialização do Estado (executado apenas na primeira chamada) ---
    if not hasattr(f, "state"):
        print("Controlador com escape: Inicializando estado...")
        # --- MODIFICADO --- Estados: "FOLLOWING", "ESCAPE_REVERSE", "ESCAPE_TURN", "ESCAPE_MOVE"
        f.state = "FOLLOWING"
        f.last_pos = None
        f.last_pos_time = 0.0
        f.stuck_timer = 0.0
        
        f.target_yaw = None
        f.escape_move_endtime = 0.0
        f.escape_reverse_endtime = 0.0 # --- NOVO --- Timer para a marcha à ré
        
        f.ultima_direita = 0.5

    # --- Constantes do Sistema de Escape ---
    STUCK_CHECK_INTERVAL = 1.0  # (s) A cada quanto tempo checamos o movimento
    STUCK_TIME_LIMIT = 5.0      # (s) Tempo parado para ser considerado "preso"
    STUCK_MOVE_THRESHOLD = 0.1  # (m) Movimento mínimo para não ser "preso"
    
    # --- NOVO: Lógica de Ré ---
    ESCAPE_REVERSE_TIME = 1.5   # (s) Tempo de marcha à ré
    ESCAPE_REVERSE_VEL = -0.2   # (m/s) Velocidade negativa (ré)
    
    ESCAPE_TURN_ANGLE = np.deg2rad(90) # (rad) Girar 90 graus à esquerda
    ESCAPE_TURN_VEL = np.deg2rad(45) # (rad/s) Velocidade angular durante o giro
    
    # --- MODIFICADO --- Aumentei o tempo de movimento para frente
    ESCAPE_MOVE_TIME = 1.5      # (s) Andar à frente por este tempo
    ESCAPE_MOVE_VEL = 0.3       # (m/s) Velocidade linear durante o escape


    # --- Constantes do Wall Following (Ajustadas) ---
    # --- MODIFICADO --- Distância menor para "grudar" mais na parede
    DISTANCIA_DESEJADA = 0.4
    KP_ANGULAR = 2
    VEL_LINEAR_FIXA = 1.0
    MAX_LINEAR_VEL = 1.0
    MAX_ANGULAR_VEL = np.deg2rad(90)
    # --- MODIFICADO --- Limite frontal menor para reagir mais perto
    LIMITE_FRONTAL = 0.6
    ZONA_MORTA_ERRO = 0.05
    DISTANCIA_SEM_PAREDE = 2.0

    # --- Obter dados atuais do robô (via 'sim') ---
    current_time = sim.getSimulationTime()
    current_pos = sim.getObjectPosition(robot, -1) # Posição [x, y, z]
    current_orient_euler = sim.getObjectOrientation(robot, -1) # Euler [a, b, g]
    current_yaw = current_orient_euler[2] # Yaw (orientação em Z)

    # --- Lógica de Detecção de "Preso" (só checa se estiver em "FOLLOWING") ---
    if f.state == "FOLLOWING":
        if f.last_pos is None:
            f.last_pos = current_pos
            f.last_pos_time = current_time
        
        time_since_last_check = current_time - f.last_pos_time
        if time_since_last_check > STUCK_CHECK_INTERVAL:
            dist_moved = np.linalg.norm(np.array(current_pos) - np.array(f.last_pos))
            
            if dist_moved < STUCK_MOVE_THRESHOLD:
                f.stuck_timer += time_since_last_check
            else:
                f.stuck_timer = 0.0
            
            f.last_pos = current_pos
            f.last_pos_time = current_time

        # --- GATILHO DO ESCAPE ---
        if f.stuck_timer > STUCK_TIME_LIMIT:
            print(f"Robô preso por {f.stuck_timer:.1f}s! Iniciando manobra de escape.")
            # --- MODIFICADO --- Inicia dando ré, não virando
            f.state = "ESCAPE_REVERSE"
            f.escape_reverse_endtime = current_time + ESCAPE_REVERSE_TIME
            f.stuck_timer = 0.0 # Reseta o contador
            

    # --- Máquina de Estados de Controle ---

    # --- NOVO --- Estado 1: Dar ré para criar espaço
    if f.state == "ESCAPE_REVERSE":
        if current_time < f.escape_reverse_endtime:
            # Dando ré
            v_cmd = ESCAPE_REVERSE_VEL
            w_cmd = 0.0
        else:
            # Ré completa, prepara para virar
            print("  Ré de escape completa. Iniciando giro.")
            f.state = "ESCAPE_TURN"
            
            # --- MODIFICADO --- Cálculo do yaw alvo movido para cá
            target_yaw_unnormalized = current_yaw + ESCAPE_TURN_ANGLE
            f.target_yaw = np.arctan2(np.sin(target_yaw_unnormalized), np.cos(target_yaw_unnormalized))
            print(f"  Yaw atual: {np.rad2deg(current_yaw):.1f} deg, Alvo: {np.rad2deg(f.target_yaw):.1f} deg")
            
            v_cmd = 0.0 # Para neste frame
            w_cmd = 0.0
            
    elif f.state == "ESCAPE_TURN":
        # --- Estado 2: Girar 90 graus à esquerda ---
        
        erro_yaw = f.target_yaw - current_yaw
        erro_yaw = np.arctan2(np.sin(erro_yaw), np.cos(erro_yaw))
        
        if abs(erro_yaw) < np.deg2rad(5):
            print("  Giro de escape completo.")
            f.state = "ESCAPE_MOVE"
            f.escape_move_endtime = current_time + ESCAPE_MOVE_TIME
            v_cmd = 0.0
            w_cmd = 0.0
        else:
            v_cmd = 0.0
            w_cmd = ESCAPE_TURN_VEL
            
    elif f.state == "ESCAPE_MOVE":
        # --- Estado 3: Mover para frente por um tempo ---
        
        if current_time < f.escape_move_endtime:
            v_cmd = ESCAPE_MOVE_VEL
            w_cmd = 0.0
        else:
            print("  Movimento de escape completo. Retornando ao Wall Following.")
            f.state = "FOLLOWING"
            f.last_pos = None
            v_cmd = 0.0
            w_cmd = 0.0
            
    elif f.state == "FOLLOWING":
        # --- Estado 0: Lógica Original (Wall Following) ---
        
        if isinstance(laser_data, dict) and "ranges" in laser_data:
            ranges = np.array(laser_data["ranges"], dtype=float)
        else:
            ranges = np.array(laser_data[:, 1], dtype=float)

        ranges = np.nan_to_num(ranges, posinf=100.0, neginf=0.0)
        num = len(ranges)
        
        if num == 0:
            v_cmd, w_cmd = 0.0, 0.0
        else:
            c = num // 2
            fatia_frente = int(num * 0.08)
            frente_ranges = ranges[c - fatia_frente : c + fatia_frente]
            dist_frente = np.min(frente_ranges) if frente_ranges.size > 0 else 100.0

            fatia_direita = int(num * 0.10)
            direita_ranges = ranges[0 : fatia_direita]
            validas = direita_ranges[direita_ranges < DISTANCIA_SEM_PAREDE]
            
            if len(validas) > 0:
                # --- MODIFICADO --- Simplificado para min(), mais robusto a "pernas"
                dist_direita = np.min(validas)
            else:
                dist_direita = DISTANCIA_SEM_PAREDE

            if dist_direita > f.ultima_direita * 1.5:
                dist_direita = f.ultima_direita * 1.1
            f.ultima_direita = dist_direita

            # --- Controle principal (copiado do seu código) ---
            
            # --- MODIFICADO: Lógica de desvio frontal ---
            if dist_frente < LIMITE_FRONTAL:
                # Controle Proporcional de Desvio:
                # Quanto mais perto (fator_proximidade -> 0), 
                # mais devagar (v_cmd -> 0) e mais rápido vira (w_cmd -> MAX)
                
                fator_proximidade = dist_frente / LIMITE_FRONTAL # (de 0.0 a 1.0)
                
                v_cmd = VEL_LINEAR_FIXA * fator_proximidade
                v_cmd = max(v_cmd, 0.05) # "Creep" (andar devagar)
                
                fator_virada = 1.0 - fator_proximidade # (de 0.0 a 1.0)
                w_cmd = MAX_ANGULAR_VEL * fator_virada # Vira à esquerda
            # --- FIM DA MODIFICAÇÃO ---
            
            elif dist_direita > DISTANCIA_SEM_PAREDE:
                v_cmd = 0.4
                w_cmd = 0.0 # Segue reto se perder a parede
            else:
                # Controle P para seguir a parede
                erro = DISTANCIA_DESEJADA - dist_direita
                if abs(erro) < ZONA_MORTA_ERRO:
                    erro = 0.0
                w_cmd = KP_ANGULAR * erro
                fator_reducao = 1.0 - (abs(w_cmd) / MAX_ANGULAR_VEL)
                v_cmd = VEL_LINEAR_FIXA * fator_reducao
                v_cmd = max(v_cmd, 0.1)

    # --- Saturação Final (para todos os estados) ---
    v_cmd = np.clip(v_cmd, -MAX_LINEAR_VEL, MAX_LINEAR_VEL)
    w_cmd = np.clip(w_cmd, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

    # --- Cinemática diferencial (para todos os estados) ---
    v_r = (2.0 * v_cmd + w_cmd * L) / (2.0 * r)
    v_l = (2.0 * v_cmd - w_cmd * L) / (2.0 * r)

    return v_l, v_r