import numpy as np
from skimage.draw import line



# conversao das coordenadas do mundo para o bloco do grid
def world_to_grid_xy(resolucao_mapa, tamanho_mapa, origem_mapa,wx, wy):
    
    ix = int((wx - origem_mapa) / resolucao_mapa)
    iy = int((wy - origem_mapa) / resolucao_mapa)
    
        
    nrows = int(tamanho_mapa / resolucao_mapa)
    ncols = int(tamanho_mapa / resolucao_mapa)
    
    if 0 <= ix < ncols and 0 <= iy < nrows:
        return ix, iy
    return None, None


# Calculo do log_odds (explicar no video)
def update_map_log_odds(log_odds_map, robot_pos, laser_data, step, last_seen,
                        l_occ=0.2, l_free=-0.7):
    
    # mexendo atualmente
    log_odds_min = -10.0
    log_odds_max = 10.0

    distancia_laiser_max = 5.0
    alcance_maximo = distancia_laiser_max * 0.99
    
    
    x, y, theta_r = robot_pos[0], robot_pos[1], robot_pos[2]
    x_Grid, y_Grid = world_to_grid_xy(x, y)
    
    if x_Grid is None or y_Grid is None:
        return log_odds_map, last_seen

    L_OCC_THRESHOLD = 3 
    
    
    
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

        beam_rows, beam_cols = rr[1:], cc[1:]

        if len(beam_rows) > 1:
            path_rows, path_cols = beam_rows[:-1], beam_cols[:-1]
            
            current_log_odds = log_odds_map[path_rows, path_cols]
            
            free_or_unknown_mask = current_log_odds < L_OCC_THRESHOLD
            
            rows_to_free = path_rows[free_or_unknown_mask]
            cols_to_free = path_cols[free_or_unknown_mask]
            log_odds_map[rows_to_free, cols_to_free] += l_free


        end_row, end_col = beam_rows[-1], beam_cols[-1]
        
        if d < alcance_maximo:
                log_odds_map[end_row, end_col] += l_occ
        else:
                log_odds_map[end_row, end_col] += l_free

        
        last_seen[rr, cc] = step

    
    np.clip(log_odds_map, log_odds_min, log_odds_max, out=log_odds_map)
    return log_odds_map, last_seen
