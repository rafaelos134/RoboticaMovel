# pos, path = mapa('../mapas_meus/mapa_convertido.png',(65, 65),(0,0),(209,208),5)

# Informacao do robotino
L = 0.135  # distancia do centro as rodas (m)
r = 0.040  # raio das rodas (m)


# pos_world = {node: pixel_to_world(x_px, y_px, 64, 10) for node, (x_px, y_px) in pos.items()}
pos_world = {node: pixel_to_world(x_px, y_px, 1259, 20) for node, (x_px, y_px) in pos.items()}
# Cinematica Inversa Holonomico
Minv =  np.array([
    [-(np.sqrt(3)/2),    1/2   ,    L],
    [      0      ,    -1    ,    L],
    [(np.sqrt(3)/2),    1/2   ,    L]
    ]) * (1 / r)


# Posicao inicial do robo - o mesmo da caminho
start_node = path[0]
start_pos_world = pos_world[start_node]
start_pos_sim = [start_pos_world[0], start_pos_world[1], 0.04]
start_ori_sim = [0, 0, np.deg2rad(90)]


# Conexao Coppelia
client = RemoteAPIClient()
sim = client.require("sim")
sim.setStepping(True)

robotname = 'robotino'
robotHandle = sim.getObject(f'/{robotname}')

wheel1 = sim.getObject(f'/{robotname}/wheel0_joint')
wheel2 = sim.getObject(f'/{robotname}/wheel1_joint')
wheel3 = sim.getObject(f'/{robotname}/wheel2_joint')


# Parar a simulação se estiver executando
initial_sim_state = sim.getSimulationState()
if initial_sim_state != 0:
    sim.stopSimulation()
    time.sleep(1)

sim.startSimulation()
sim.setObjectPosition(robotHandle, sim.handle_world, start_pos_sim)
sim.setObjectOrientation(robotHandle, sim.handle_world, start_ori_sim)
sim.step()

# Estado inicial
q = np.array([start_pos_sim[0], start_pos_sim[1], start_ori_sim[2]])

print("Iniciando o caminho")

# Caminhos a ser seguidos
waypoints = [pos_world[node] for node in path]
target_index = 0
tolerance = 0.2

while target_index < len(waypoints):
    target = np.array(waypoints[target_index])
    pos_sim = sim.getObjectPosition(robotHandle, sim.handle_world)
    q[:2] = pos_sim[:2]

    # Erro no referencial do mundo
    erro = target - q[:2]
    dist = np.linalg.norm(erro)

    # Controlador proporcional
    vx_world = 0.7 * erro[0]
    vy_world = 0.7 * erro[1]

    # Limita velocidades
    vx_world = np.clip(vx_world, -0.3, 0.3)
    vy_world = np.clip(vy_world, -0.3, 0.3)

    # Converte velocidade do mundo → corpo (mantendo orientação fixa)
    v_body = Rz(-start_ori_sim[2]) @ np.array([vx_world, vy_world, 0.04])


    # vz_body e descartado por ser 0
    vx_body, vy_body , vz_body = v_body
    w = 0.0  # sem rotação

    # Calcula velocidades das rodas
    qdot = np.array([vx_body, vy_body, w])

    
    u = Minv @ qdot  # usa cinemática inversa para gerar rodas

    # Envia velocidades
    sim.setJointTargetVelocity(wheel1, u[0])
    sim.setJointTargetVelocity(wheel2, u[1])
    sim.setJointTargetVelocity(wheel3, u[2])

    # Avança um passo
    sim.step()

    # Verifica se chegou
    if dist < tolerance:
        print(f"Chegou em {path[target_index]} -> indo para o próximo ponto")
        target_index += 1


print("Caminho concluído! Parando robô...")
for w in [wheel1, wheel2, wheel3]:
    sim.setJointTargetVelocity(w, 0)

pos_final = sim.getObjectPosition(robotHandle, sim.handle_world)
print("Posição final (SIM):", pos_final)
sim.stopSimulation()
print("Program ended.")
