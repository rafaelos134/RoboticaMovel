import numpy as np
from skimage.draw import line




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
