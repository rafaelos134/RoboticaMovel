import numpy as np


class Controlador:
    
    def __init__(self):
        self.state = "SEGUINDO"
        self.last_posi = None
        self.last_time = 0.0
        self.tempo_preso = 0.0
        
        self.alvo_yaw = None
        self.tempo_em_frente = 0.0
        self.tempo_escape_re = 0.0
        
        self.ultima_direita = 0.5
        
        
    def controle_wall_following_com_escape(self, sim, robot, laser_data, r, L):
        
        
        # CTE limites
        max_vel_linear = 1.0
        max_vel_angular = np.deg2rad(90)
        
        # CTE escape
        intervalo_checagem = 1.0
        movimento_min = 0.1     # movimento para nao ser considerado preso
        tempo_limite = 5.0      # tempo para ser considerado preso
        
        tempo_re = 1.5
        vel_re = -0.2
        
        tempo_angulo = np.deg2rad(90)
        vel_angulo = np.deg2rad(45)

        tempo_em_frente = 1.5   
        vel_em_frente = 0.3      

        # CTE seguindo parede
        distancia_desejada = 0.4
        ganho_angular = 2
        vel_linear_fixa = 1.0
        
        # --- MODIFICADO --- Limite frontal menor para reagir mais perto
        limite_frontal = 0.6
        zona_erro = 0.05
        distancia_sem_parede = 2.0
        
        
        
        
    
        # Pegando informações do robo via sim
        time = sim.getSimulationTime()
        posi = sim.getObjectPosition(robot, -1)
        ori  = sim.getObjectOrientation(robot, -1)
        yaw  = ori[2] 

        # Verificacao se esta preso
        if self.state == "SEGUINDO":
            
            if self.last_posi is None:
                self.last_posi = posi
                self.last_time = time
            
            ultima_veirificao = time - self.last_time
            
            if ultima_veirificao > intervalo_checagem:
                dist_moved = np.linalg.norm(np.array(posi) - np.array(self.last_posi))
                
                if dist_moved < movimento_min:
                    self.tempo_preso += ultima_veirificao
                else:
                    self.tempo_preso = 0.0
                
                self.last_posi = posi
                self.last_time = time

            
            if self.tempo_preso > tempo_limite:
                print(f"Ativando Ré")
                self.state = "RE"
                self.tempo_escape_re = time + tempo_re
                self.tempo_preso = 0.0
                

        # inicioando a re
        if self.state == "RE":
            if time < self.tempo_escape_re:
                v_cmd = vel_re
                w_cmd = 0.0
            else:
                print("Ré de escape completa. Iniciando giro.")
                self.state = "GIRO"
                
                alvo_yaw_semNormalizaco = yaw + tempo_angulo
                self.alvo_yaw = np.arctan2(np.sin(alvo_yaw_semNormalizaco), np.cos(alvo_yaw_semNormalizaco))
                
                # robo para
                v_cmd = 0.0
                w_cmd = 0.0
                
        
        # rotaciona 90 graus a esquerda        
        elif self.state == "GIRO":
            
            erro_yaw = self.alvo_yaw - yaw
            erro_yaw = np.arctan2(np.sin(erro_yaw), np.cos(erro_yaw))
            
            if abs(erro_yaw) < np.deg2rad(5):
                print("Giro de escape completo.")
                self.state = "EM_FRENTE"
                self.tempo_em_frente = time + tempo_em_frente
                v_cmd = 0.0
                w_cmd = 0.0
            else:
                v_cmd = 0.0
                w_cmd = vel_angulo

        # seguindo em Frente
        elif self.state == "EM_FRENTE":
            
            if time < self.tempo_em_frente:
                v_cmd = vel_em_frente
                w_cmd = 0.0
            else:
                print("Movimento de escape completo. Retornando ao Wall Following.")
                self.state = "SEGUINDO"
                self.last_posi = None
                v_cmd = 0.0
                w_cmd = 0.0
                
                
                
        # Seguindo parede        
        elif self.state == "SEGUINDO":
            
            # # verificar qual o caso correto
            # if isinstance(laser_data, dict) and "ranges" in laser_data:
            #     ranges = np.array(laser_data["ranges"], dtype=float)
            # else:
            #     ranges = np.array(laser_data[:, 1], dtype=float)
            ranges = np.array(laser_data["ranges"], dtype=float)
            ranges = np.nan_to_num(ranges, posinf=100.0, neginf=0.0)
            num = len(ranges)
            
            if num == 0:
                v_cmd, w_cmd = 0.0, 0.0
            else:
                c = num // 2 # Meio da visao
                
                fatia_frente = int(num * 0.08)
                frente_ranges = ranges[c - fatia_frente : c + fatia_frente]
                dist_frente = np.min(frente_ranges) if frente_ranges.size > 0 else 100.0

                fatia_direita = int(num * 0.10)
                direita_ranges = ranges[0 : fatia_direita]
                validas = direita_ranges[direita_ranges < distancia_sem_parede]
                
                if len(validas) > 0:
                    dist_direita = np.min(validas)
                else:
                    dist_direita = distancia_sem_parede
                
                # limpeza de ruido na idendificao da parede
                if dist_direita > self.ultima_direita * 1.5:
                    dist_direita = self.ultima_direita * 1.1
                self.ultima_direita = dist_direita



                if dist_frente < limite_frontal:
                    
                    fator_proximidade = dist_frente / limite_frontal # normalizacao
                    
                    v_cmd = vel_linear_fixa * fator_proximidade
                    v_cmd = max(v_cmd, 0.05)
                    
                    fator_virada = 1.0 - fator_proximidade 
                    w_cmd = max_vel_angular * fator_virada 
                
                
                elif dist_direita > distancia_sem_parede:
                    v_cmd = 0.4
                    w_cmd = 0.0 
                else:
                    erro = distancia_desejada - dist_direita
                    if abs(erro) < zona_erro:
                        erro = 0.0
                    w_cmd = ganho_angular * erro
                    fator_reducao = 1.0 - (abs(w_cmd) / max_vel_angular)
                    v_cmd = vel_linear_fixa * fator_reducao
                    v_cmd = max(v_cmd, 0.1)


        # Limitador de velocidade
        v_cmd = np.clip(v_cmd, -max_vel_linear, max_vel_linear)
        w_cmd = np.clip(w_cmd, -max_vel_angular, max_vel_angular)

        # Cinematica inversa -> Nao Holonomico
        w_r = (v_cmd/r) + ((w_cmd * L) / (2.0 * r))
        w_l = (v_cmd/r) - ((w_cmd * L) / (2.0 * r))
        
        
        return w_l, w_r
            
