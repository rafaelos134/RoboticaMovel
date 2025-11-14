import numpy as np


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