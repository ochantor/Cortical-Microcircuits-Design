import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
try:
    import winsound
except ImportError:
    winsound = None
from matplotlib.patches import Ellipse
import sys


N = 25
ANGULOS = np.linspace(0, 2*np.pi, N, endpoint=False)
TIME_WARP = 1.0

# --- Parametros de relajacion softmax (reemplazan a "decay + argmax") ---
# alpha: que tan rapido se relaja la actividad hacia el objetivo cada frame.
# temperatura: que tan afilada es la competencia (temperatura baja = casi
# un solo ganador; temperatura alta = varias unidades comparten actividad).
ALPHA_MOT, T_MOT = 0.35, 0.25
ALPHA_NAV, T_NAV = 0.40, 0.15
ALPHA_N2,  T_N2  = 0.35, 0.20
ALPHA_N3,  T_N3  = 0.35, 0.20

# --- N+3: histeresis de amenaza (memoria de peligro) ---
# En vez de un "danger" que solo crece con el tiempo (sin memoria real
# del evento), N+3 mantiene un estado de alerta que sube rapido al
# percibir al depredador cerca y decae LENTO con constante de tiempo TAU_N3.
# Esto es lo que evita que la criatura sea sorprendida por el "depredador
# fantasma": aunque el depredador salga del radio de percepcion, la
# alerta persiste por ~TAU_N3 frames antes de apagarse.
PRED_PERCEPTION_RADIUS = 0.45   # radio dentro del cual el depredador es "percibido"
TAU_N3 = 16.0                   # tau del paper: constante de decaimiento de la alerta (frames)
K_ALERTA_SUBIDA = 0.9            # velocidad de subida de la alerta al percibir amenaza

# --- Maduracion (reemplaza a build_despierto) ---
EDAD_DESPIERTE = 0.6
RHO_MADURACION = 12.0   # pendiente de la sigmoide de maduracion

# --- Carga continua (reemplaza a tiene_brizna) ---
PICKUP_RADIUS = 0.12
NIDO_RADIUS = 0.25
K_PICKUP = 0.55  # tasa de carga por frame cuando esta sobre el material
K_DROP = 0.45     # tasa de descarga por frame cuando esta sobre el nido
STEEPNESS_GATE = 50.0  # que tan abrupta es la ventana de "estoy encima"

IMPULSO_MINIMO = 0.15

def _compuerta(x, filo=30.0):
    z = np.clip(-filo * x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))

def relajar_softmax(actividad, energias, alpha, temperatura):
    """Competencia continua sin argmax: la actividad se relaja hacia una
    distribucion softmax de las energias. Es la aproximacion de campo medio
    estandar del equilibrio de una red de inhibicion lateral tipo Amari
    (1977) -- no resuelve la EDO completa paso a paso, pero converge al
    mismo tipo de distribucion de equilibrio (una region gana, el resto
    decae), sin que ningun paso del codigo pregunte "cual es el maximo"."""
    e = energias - np.max(energias)
    pesos = np.exp(e / temperatura)
    objetivo = pesos / np.sum(pesos)
    return (1.0 - alpha) * actividad + alpha * objetivo

# ============================================================
# ESTADO DEL MUNDO
# ============================================================

pos = np.array([0.0, 0.0])
theta = 0.0
hunger = 0.3
safety = 0.0
danger = 0.0
border_stress = 0.0
alerta_n3 = 0.0   # estado de alerta con histeresis (N+3)

food_pos = np.array([0.65, 0.45]); food_theta = 0.0; food_radius = 0.65
home_pos = np.array([-0.55, -0.35]); home_theta = np.pi; home_radius = 0.60
pred_pos = np.array([0.0, -0.7]); pred_theta = 0.0; pred_radius = 0.8

nido_pos = np.array([-0.70, 0.70])
nido_tamaño = 0.30
nido_celdas = 3
tam_celda = nido_tamaño / nido_celdas

materiales_maximos = 9
nido_completado = False
total_depositado = 0.0        # integral continua de flujo de descarga
total_recogido_frac = 0.0     # integral continua de flujo de carga (para consumir el objeto)

material_pos = np.array([0.0, 0.0])
material_activo = False

# --- Reemplaza a tiene_brizna: variable continua de carga ---
L = 0.0

edad = 0.0
instinto_construccion = 0.0

impulso_construir = 0.8
tasa_impulso_base = 0.015
urgencia_constructiva = 1.2

materiales_generados = 0
max_materiales_generados = 30
tiempo_sin_material = 0.0
tiempo_sin_construir = 0.0

food_lock = False
home_lock = False

# ============================================================
# TRES AREAS CORTICALES (misma estructura de pesos que antes)
# ============================================================

activity_mot = np.ones(N) / N
cms_mot = []
for k in range(N):
    cms_mot.append({
        "angle": ANGULOS[k],
        "food_weight": np.random.uniform(1.2, 1.6),
        "home_weight": np.random.uniform(1.3, 1.7),
        "pred_weight": np.random.uniform(1.2, 1.7),
        "explore": np.random.uniform(0, 0.06)
    })

activity_nav = np.ones(N) / N
cms_nav = []
for k in range(N):
    cms_nav.append({
        "angle": ANGULOS[k],
        "border_weight": np.random.uniform(0.8, 1.2),
        "explore": np.random.uniform(0, 0.05)
    })

activity_n2 = np.ones(N) / N
cms_n2 = []
for k in range(N):
    cms_n2.append({
        "angle": ANGULOS[k],
        "material_weight": np.random.uniform(1.2, 1.6),
        "nido_weight": np.random.uniform(1.3, 1.7),
        "explore": np.random.uniform(0, 0.04)
    })

# --- AREA N+3: evasion con memoria de amenaza (histeresis) ---
activity_n3 = np.ones(N) / N
cms_n3 = []
for k in range(N):
    cms_n3.append({
        "angle": ANGULOS[k],
        "escape_weight": np.random.uniform(1.2, 1.7),
        "explore": np.random.uniform(0, 0.04)
    })

# ============================================================
# MAPA VISUAL CORTICAL
# ============================================================

brain = np.zeros((21, 21))
# Layout 2x2 compacto, corrido hacia abajo para no chocar con el texto
# de variables (que ocupa la franja superior, filas 0-6):
#   MOT (fila 7-11, izq)   NAV (fila 7-11, der)
#   N+3 (fila 13-17, izq)  N+2 (fila 13-17, der)
core_mot = [(r, c) for r in range(7, 12) for c in range(5, 10)]
core_nav = [(r, c) for r in range(7, 12) for c in range(12, 17)]
core_n3  = [(r, c) for r in range(13, 18) for c in range(5, 10)]
core_n2  = [(r, c) for r in range(13, 18) for c in range(12, 17)]

# ============================================================
# FIGURA
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1)
ax1.set_facecolor("black")
ax1.set_title("MUNDO - VERSION CONTINUA (sin argmax, sin booleanos de control) + N+3 (histeresis de amenaza)", color='white', fontsize=9)

home_zone = plt.Circle((home_pos[0], home_pos[1]), 0.25, color='blue', alpha=0.15, fill=True)
ax1.add_patch(home_zone)
home_safety_zone = plt.Circle((home_pos[0], home_pos[1]), 0.30, color='blue', alpha=0.05, fill=True)
ax1.add_patch(home_safety_zone)
boundary = plt.Rectangle((-0.95, -0.95), 1.9, 1.9, edgecolor='red', linestyle='--', fill=False, alpha=0.3)
ax1.add_patch(boundary)

dot, = ax1.plot([0], [0], 'wo', markersize=7)
food_dot, = ax1.plot(food_pos[0], food_pos[1], 'r*', markersize=14)
home_dot, = ax1.plot(home_pos[0], home_pos[1], 'bo', markersize=12)
pred_dot, = ax1.plot(pred_pos[0], pred_pos[1], 'go', markersize=14)
material_patch = Ellipse((0, 0), width=0.08, height=0.05, facecolor='yellow', edgecolor='yellow', alpha=0.9)
ax1.add_patch(material_patch)
material_patch.set_visible(False)
line, = ax1.plot([0, 0], [0, 0], 'w-', linewidth=2)

inicio_x = nido_pos[0] - nido_tamaño / 2
inicio_y = nido_pos[1] - nido_tamaño / 2
ax1.plot([inicio_x, inicio_x + nido_tamaño], [inicio_y, inicio_y], color='yellow', linewidth=2, alpha=0.8)
ax1.plot([inicio_x, inicio_x + nido_tamaño], [inicio_y + nido_tamaño, inicio_y + nido_tamaño], color='yellow', linewidth=2, alpha=0.8)
ax1.plot([inicio_x, inicio_x], [inicio_y, inicio_y + nido_tamaño], color='yellow', linewidth=2, alpha=0.8)
ax1.plot([inicio_x + nido_tamaño, inicio_x + nido_tamaño], [inicio_y, inicio_y + nido_tamaño], color='yellow', linewidth=2, alpha=0.8)
for i in range(1, nido_celdas):
    x = inicio_x + i * tam_celda
    ax1.plot([x, x], [inicio_y, inicio_y + nido_tamaño], color='yellow', linewidth=1, alpha=0.5)
    y = inicio_y + i * tam_celda
    ax1.plot([inicio_x, inicio_x + nido_tamaño], [y, y], color='yellow', linewidth=1, alpha=0.5)

celdas = []
for fila in range(3):
    for col in range(3):
        x = inicio_x + col * tam_celda
        y = inicio_y + fila * tam_celda
        celda = plt.Rectangle((x, y), tam_celda, tam_celda, facecolor='yellow', alpha=0.0, edgecolor='none')
        ax1.add_patch(celda)
        celdas.append(celda)

img = ax2.imshow(brain, vmin=0, vmax=1, cmap='inferno')
ax2.axvline(10.5, color='white', linestyle=':', alpha=0.2, linewidth=0.5)
ax2.axhline(12, color='white', linestyle=':', alpha=0.2, linewidth=0.5)
ax2.text(7, 12.4, 'MOT', color='white', fontsize=8, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.7))
ax2.text(14, 12.4, 'NAV', color='white', fontsize=8, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='darkblue', alpha=0.7))
ax2.text(7, 18.4, 'N+3', color='white', fontsize=8, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='darkorange', alpha=0.7))
ax2.text(14, 18.4, 'N+2', color='white', fontsize=8, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.7))
ax2.set_title("CORTEZA (competencia softmax continua)", color='white', fontsize=9)
info_text = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, color='white', fontsize=8,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def actualizar_nido():
    global celdas
    celdas_llenas = min(int(np.floor(total_depositado)), materiales_maximos)
    for celda in celdas:
        celda.set_alpha(0.0)
    for i in range(celdas_llenas):
        celdas[i].set_alpha(0.6)

def generar_material():
    global material_pos, material_activo, materiales_generados
    if materiales_generados >= max_materiales_generados or nido_completado:
        material_activo = False
        return
    if material_activo:
        return
    centro_x, centro_y, radio = 0.75, -0.75, 0.20
    for _ in range(200):
        angulo = np.random.uniform(0, 2*np.pi)
        distancia = np.random.uniform(0, radio)
        nueva_pos = np.array([centro_x + distancia*np.cos(angulo), centro_y + distancia*np.sin(angulo)])
        if np.linalg.norm(nueva_pos - nido_pos) > 0.20 and np.linalg.norm(nueva_pos - home_pos) > 0.20:
            material_pos = nueva_pos
            material_activo = True
            materiales_generados += 1
            return
    material_activo = False

def reset_system():
    global pos, theta, hunger, safety, danger, border_stress, alerta_n3
    global activity_mot, activity_nav, activity_n2, activity_n3, brain
    global food_pos, home_pos, pred_pos, food_theta, home_theta, pred_theta
    global food_lock, home_lock
    global material_activo, nido_completado
    global impulso_construir, materiales_generados, tiempo_sin_material
    global tiempo_sin_construir, urgencia_constructiva
    global edad, instinto_construccion, L, total_depositado, total_recogido_frac

    try:
        winsound.Beep(180, 600)
    except Exception:
        pass

    pos = np.array([0.0, 0.0]); theta = 0.0
    hunger = 0.3; safety = 0.0; danger = 0.0; border_stress = 0.0; alerta_n3 = 0.0
    activity_mot = np.ones(N)/N; activity_nav = np.ones(N)/N; activity_n2 = np.ones(N)/N; activity_n3 = np.ones(N)/N
    brain = np.zeros((21, 21))

    food_theta = np.random.uniform(0, 2*np.pi)
    home_theta = np.random.uniform(0, 2*np.pi)
    pred_theta = np.random.uniform(0, 2*np.pi)
    food_pos = np.array([food_radius*np.cos(food_theta), 0.55*np.sin(1.7*food_theta)])
    home_pos = np.array([home_radius*np.cos(home_theta), 0.45*np.sin(1.3*home_theta)])
    pred_pos = np.array([pred_radius*np.sin(pred_theta), pred_radius*np.cos(pred_theta)])

    food_lock = False; home_lock = False
    material_activo = False; nido_completado = False
    L = 0.0; total_depositado = 0.0; total_recogido_frac = 0.0
    impulso_construir = 0.8
    materiales_generados = 0; tiempo_sin_material = 0.0
    tiempo_sin_construir = 0.0
    urgencia_constructiva = 1.2
    edad = 0.0; instinto_construccion = 0.0
    actualizar_nido()

# ============================================================
# UPDATE LOOP
# ============================================================

def update(frame):
    global pos, theta, hunger, safety, danger, border_stress, alerta_n3
    global activity_mot, activity_nav, activity_n2, activity_n3, brain
    global food_pos, home_pos, pred_pos, food_theta, home_theta, pred_theta
    global food_lock, home_lock
    global material_activo, material_pos, nido_completado
    global impulso_construir, materiales_generados, tiempo_sin_material
    global tiempo_sin_construir, urgencia_constructiva
    global edad, instinto_construccion, L, total_depositado, total_recogido_frac

    dt = TIME_WARP

    # --- 1. distancias ---
    dist_food = np.linalg.norm(pos - food_pos)
    dist_home = np.linalg.norm(pos - home_pos)
    dist_pred = np.linalg.norm(pos - pred_pos)
    dist_nido = np.linalg.norm(pos - nido_pos)
    dist_material = np.linalg.norm(pos - material_pos) if material_activo else 999.0

    # --- 2. homeostasis ---
    hunger = np.clip(hunger + 0.0015*dt, 0, 1)
    safety = np.clip(safety + (0.0 if dist_home < 0.25 else 0.0020*dt) - (safety if dist_home < 0.25 else 0.0), 0, 1)
    if dist_home < 0.25:
        safety = 0.0
    danger = np.clip(danger + 0.003*dt, 0, 1)

    # --- gating homeostatico continuo (igual que en la version que funciona) ---
    compuerta_en_casa = _compuerta(0.25 - dist_home)
    compuerta_poco_hambre = _compuerta(0.40 - hunger)
    compuerta_seguridad_alta = _compuerta(safety - 0.80)
    compuerta_urgencia = _compuerta(hunger - 0.50) * _compuerta(0.40 - safety)

    atenuacion = np.clip((1.0 - safety) / 0.20, 0, 1)
    eh_atenuado = hunger * atenuacion
    sn_atenuado = safety * 1.8
    eh_urgente = hunger * 1.5
    sn_urgente = safety * 0.3
    eh_plano = hunger
    sn_plano = safety

    peso_atenuado = compuerta_seguridad_alta
    peso_urgente = compuerta_urgencia * (1.0 - peso_atenuado)
    peso_plano = np.clip(1.0 - peso_atenuado - peso_urgente, 0, 1)

    effective_hunger_lejos = peso_atenuado*eh_atenuado + peso_urgente*eh_urgente + peso_plano*eh_plano
    current_safety_need_lejos = peso_atenuado*sn_atenuado + peso_urgente*sn_urgente + peso_plano*sn_plano

    en_casa_activo = compuerta_en_casa * (1.0 - compuerta_poco_hambre)
    reposo_intensidad = compuerta_en_casa * compuerta_poco_hambre
    lejos = 1.0 - compuerta_en_casa

    effective_hunger = en_casa_activo*hunger + lejos*effective_hunger_lejos
    current_safety_need = reposo_intensidad*1.0 + lejos*current_safety_need_lejos

    # --- 3. edad e instinto: sigmoide continua, sin bandera booleana ---
    edad += 0.01 * dt
    instinto_construccion = _compuerta(edad - EDAD_DESPIERTE, filo=RHO_MADURACION)

    # --- 4. urgencia constructiva (bookkeeping de progreso, no de control motor) ---
    if not nido_completado:
        urgencia_constructiva += 0.005*dt
        urgencia_constructiva += 0.015*dt * _compuerta(dist_material - 999 + 1) * (1.0 - L)  # aporte suave si hay material visible y no se carga aun
        urgencia_constructiva += 0.02*dt * L
        progreso = total_depositado / materiales_maximos
        urgencia_constructiva += 0.005*progreso*dt
        tiempo_sin_construir += dt
        urgencia_constructiva += 0.01*dt * _compuerta(tiempo_sin_construir - 60, filo=0.3)
        urgencia_constructiva = np.clip(urgencia_constructiva, 0, 1.5)

    # --- 5. estres de bordes ---
    dist_to_wall_x = 1.0 - abs(pos[0])
    dist_to_wall_y = 1.0 - abs(pos[1])
    closest_wall_dist = min(dist_to_wall_x, dist_to_wall_y)
    border_stress = (np.clip((0.25 - closest_wall_dist)/0.25, 0, 1) ** 2) if closest_wall_dist < 0.25 else 0.0

    # --- 6. movimiento de entidades ---
    food_theta += 0.015*dt
    food_pos = np.array([food_radius*np.cos(food_theta), 0.55*np.sin(1.7*food_theta)])
    home_theta -= 0.010*dt
    home_pos = np.array([home_radius*np.cos(home_theta), 0.45*np.sin(1.3*home_theta)])
    home_zone.set_center((home_pos[0], home_pos[1]))
    home_safety_zone.set_center((home_pos[0], home_pos[1]))

    # --- 7. depredador ---
    pred_speed = 0.023*dt
    if dist_home < 0.30:
        pred_theta += np.random.uniform(-0.8, 0.8)
    else:
        to_prey = pos - pred_pos
        pred_theta = np.arctan2(to_prey[1], to_prey[0])
    pred_pos = pred_pos + pred_speed*np.array([np.cos(pred_theta), np.sin(pred_theta)])
    pred_pos = np.clip(pred_pos, -1.2, 1.2)

    # ========================================================
    # 7b. N+3: PERCEPCION Y ALERTA CON HISTERESIS
    # ========================================================
    # percepcion_amenaza: ventana suave (sigmoide) de "el depredador esta
    # dentro del radio de percepcion", no un booleano de umbral duro.
    percepcion_amenaza = _compuerta(PRED_PERCEPTION_RADIUS - dist_pred, filo=25.0)

    # Integrador de fuga con asimetria temporal explicita:
    #   - subida: rapida, proporcional a la percepcion actual (reaccion inmediata)
    #   - bajada: lenta, con constante de tiempo TAU_N3 (memoria del peligro)
    # Esto es justo lo que el baseline sin N+3 no tenia: ahi "danger" solo
    # crecia con el tiempo y nunca respondia realmente al evento de
    # percibir al depredador, asi que la criatura no tenia forma de
    # sostener cautela despues de que el depredador desaparecia de vista
    # (el "depredador fantasma" la sorprendia sistematicamente).
    subida_alerta = K_ALERTA_SUBIDA * percepcion_amenaza * (1.0 - alerta_n3)
    bajada_alerta = (alerta_n3 / TAU_N3) * (1.0 - percepcion_amenaza)
    alerta_n3 = np.clip(alerta_n3 + dt*(subida_alerta - bajada_alerta), 0, 1)

    # --- 8. colisiones (eventos objetivos del mundo, no del cerebro) ---
    if dist_food < 0.10 and not food_lock:
        hunger = 0.0
        try: winsound.Beep(1200, 120)
        except Exception: pass
        food_lock = True
    if dist_home < 0.10 and not home_lock:
        try: winsound.Beep(500, 180)
        except Exception: pass
        home_lock = True
    if dist_pred < 0.12:
        if dist_home < 0.30:
            danger = 0.0
        else:
            reset_system()
            return dot, line, img, food_dot, home_dot, pred_dot, material_patch, info_text
    if dist_food > 0.18: food_lock = False
    if dist_home > 0.18: home_lock = False

    # --- 9. impulso constructivo (bookkeeping continuo, ya sin piso artificial) ---
    if nido_completado:
        impulso_construir *= 0.95
    else:
        impulso_construir += tasa_impulso_base*dt
        impulso_construir += urgencia_constructiva*0.02*dt
        impulso_construir += 0.01*dt * (1.0 - L) * (1.0 if material_activo else 0.0)
        impulso_construir += 0.02*dt * L
        progreso = total_depositado / materiales_maximos
        impulso_construir += 0.005*progreso*dt
        impulso_construir += 0.01*dt * _compuerta(tiempo_sin_construir - 50, filo=0.3)
    impulso_construir = np.clip(impulso_construir, 0, 1)

    # ========================================================
    # 10. CARGA CONTINUA L(t) -- reemplaza a tiene_brizna
    # ========================================================
    # g_pickup/g_drop: "estoy sobre el material" / "estoy sobre el nido",
    # como ventanas suaves (sigmoides), no como comparaciones booleanas
    # aisladas de la dinamica.
    g_pickup = _compuerta(PICKUP_RADIUS - dist_material, filo=STEEPNESS_GATE) if material_activo else 0.0
    g_drop = _compuerta(NIDO_RADIUS - dist_nido, filo=STEEPNESS_GATE) if not nido_completado else 0.0

    flujo_entrada = K_PICKUP * g_pickup * (1.0 - L)
    flujo_salida = K_DROP * g_drop * L
    L = np.clip(L + dt*(flujo_entrada - flujo_salida), 0, 1)

    # El conteo de "cuanto se ha entregado" es la integral del MISMO flujo
    # que vacia L -- no una deteccion de umbral aparte.
    total_depositado = min(materiales_maximos, total_depositado + dt*flujo_salida)
    total_recogido_frac += dt*flujo_entrada
    if total_recogido_frac >= 1.0 and material_activo:
        # se termino de "cargar" una unidad completa de material: el objeto
        # discreto del mundo se consume (evento del entorno, no del cerebro)
        material_activo = False
        total_recogido_frac -= 1.0
        try: winsound.Beep(800, 100)
        except Exception: pass

    if total_depositado >= materiales_maximos:
        nido_completado = True
        material_activo = False
    actualizar_nido()

    # --- 11. generar nuevo material ---
    if not nido_completado and not material_activo:
        if materiales_generados < max_materiales_generados:
            generar_material()
        else:
            tiempo_sin_material += dt
            if tiempo_sin_material > 150:
                materiales_generados = 0
                tiempo_sin_material = 0.0
                generar_material()

    # --- 12. direcciones sensoriales ---
    vec_food = food_pos - pos; vec_home = home_pos - pos; vec_pred = pred_pos - pos
    angle_food = np.arctan2(vec_food[1], vec_food[0])
    angle_home = np.arctan2(vec_home[1], vec_home[0])
    angle_pred = np.arctan2(vec_pred[1], vec_pred[0])

    fuerza_oeste = 1.0/(1.0 + (pos[0]-(-0.95)))
    fuerza_este = 1.0/(1.0 + (0.95-pos[0]))
    fuerza_sur = 1.0/(1.0 + (pos[1]-(-0.95)))
    fuerza_norte = 1.0/(1.0 + (0.95-pos[1]))
    vec_border = np.array([fuerza_este-fuerza_oeste, fuerza_norte-fuerza_sur])
    angle_border = np.arctan2(vec_border[1], vec_border[0])

    # ========================================================
    # 13. AREA 1 (MOT): energias -> relajacion softmax (sin argmax)
    # ========================================================
    # N+3 operando: el termino de evitacion de MOT ya no usa el "danger"
    # que solo crecia con el tiempo -- usa la alerta con histeresis, que
    # sube al percibir al depredador y persiste ~TAU_N3 frames despues.
    current_danger_factor = 0.0 if dist_home < 0.25 else alerta_n3
    angs = np.array([cm["angle"] for cm in cms_mot])
    fw = np.array([cm["food_weight"] for cm in cms_mot])
    hw = np.array([cm["home_weight"] for cm in cms_mot])
    pw = np.array([cm["pred_weight"] for cm in cms_mot])
    ex = np.array([cm["explore"] for cm in cms_mot])
    energies_mot = (effective_hunger*fw*np.cos(angle_food-angs)
                    + current_safety_need*hw*np.cos(angle_home-angs)
                    - current_danger_factor*pw*np.cos(angle_pred-angs)
                    + ex*np.random.uniform(-1, 1, N)
                    + 0.06*np.random.randn(N))
    activity_mot = relajar_softmax(activity_mot, energies_mot, ALPHA_MOT, T_MOT)

    # ========================================================
    # 14. AREA 2 (NAV): idem, con destello de reposo mezclado continuamente
    # ========================================================
    angs_n = np.array([cm["angle"] for cm in cms_nav])
    bw = np.array([cm["border_weight"] for cm in cms_nav])
    exn = np.array([cm["explore"] for cm in cms_nav])
    energies_nav = -border_stress*bw*np.cos(angle_border-angs_n) + exn*np.random.uniform(-1, 1, N)
    ruido_reposo = np.random.uniform(-1, 1, N) * 2.0
    energies_nav_eff = (1.0 - reposo_intensidad)*energies_nav + reposo_intensidad*ruido_reposo
    activity_nav = relajar_softmax(activity_nav, energies_nav_eff, ALPHA_NAV, T_NAV)

    # ========================================================
    # 15. AREA 3 (N+2): la formula ya estaba escrita como interpolacion
    # -- ahora L es continuo de verdad, no 0/1
    # ========================================================
    vec_material = (material_pos - pos) if material_activo else np.array([0.0, 0.0])
    vec_nido = nido_pos - pos
    angle_material = np.arctan2(vec_material[1], vec_material[0]) if np.linalg.norm(vec_material) > 0 else 0.0
    angle_nido = np.arctan2(vec_nido[1], vec_nido[0])
    material_disponible = 1.0 if material_activo else 0.0

    angs2 = np.array([cm["angle"] for cm in cms_n2])
    mw = np.array([cm["material_weight"] for cm in cms_n2])
    nw = np.array([cm["nido_weight"] for cm in cms_n2])
    ex2 = np.array([cm["explore"] for cm in cms_n2])
    material_align = np.cos(angle_material-angs2) if material_activo else np.zeros(N)
    nido_align = np.cos(angle_nido-angs2)

    energies_n2 = (
        (1-L)*material_disponible*mw*material_align*(0.8+0.5*instinto_construccion)
        + L*nw*nido_align*(0.7+0.5*instinto_construccion)
        + L*mw*material_align*0.1
        + (1-L)*material_disponible*nw*nido_align*0.1
        + (1-L)*(1-material_disponible)*ex2*np.random.uniform(0.5, 1.5, N)*(0.3+instinto_construccion)
        + ex2*np.random.uniform(-1, 1, N)
    )
    activity_n2 = relajar_softmax(activity_n2, energies_n2, ALPHA_N2, T_N2)

    # ========================================================
    # 15b. AREA 4 (N+3): energias de evasion, activas solo mientras
    # alerta_n3 > 0 -- la propia alerta con histeresis es la que decide
    # cuanto tiempo esta area sigue "encendida" despues de perder de
    # vista al depredador.
    # ========================================================
    angle_evasion = angle_pred + np.pi  # direccion opuesta al depredador
    angs3 = np.array([cm["angle"] for cm in cms_n3])
    ew = np.array([cm["escape_weight"] for cm in cms_n3])
    ex3 = np.array([cm["explore"] for cm in cms_n3])
    evasion_align = np.cos(angle_evasion - angs3)

    energies_n3 = (
        alerta_n3 * ew * evasion_align
        + ex3 * np.random.uniform(-1, 1, N)
        + 0.06 * np.random.randn(N)
    )
    activity_n3 = relajar_softmax(activity_n3, energies_n3, ALPHA_N3, T_N3)

    # ========================================================
    # 16. SINTESIS MOTORA (poblacional, igual estructura que siempre)
    # ========================================================
    vecs = np.stack([np.cos(ANGULOS), np.sin(ANGULOS)], axis=1)
    motor_mot = activity_mot @ vecs
    motor_nav = activity_nav @ vecs
    motor_n2 = activity_n2 @ vecs
    motor_n3 = activity_n3 @ vecs
    if np.linalg.norm(motor_mot) > 0: motor_mot = motor_mot/np.linalg.norm(motor_mot)
    if np.linalg.norm(motor_nav) > 0: motor_nav = motor_nav/np.linalg.norm(motor_nav)
    if np.linalg.norm(motor_n2) > 0: motor_n2 = motor_n2/np.linalg.norm(motor_n2)
    if np.linalg.norm(motor_n3) > 0: motor_n3 = motor_n3/np.linalg.norm(motor_n3)

    # ========================================================
    # 17. INTEGRACION -- todo continuo, sin "if build_despierto"
    # ========================================================
    g_impulso = _compuerta(impulso_construir - IMPULSO_MINIMO, filo=40.0)
    peso_n2 = instinto_construccion * g_impulso * impulso_construir * 0.7
    factor_hambre = 1.0 - hunger*0.5
    peso_n2 *= factor_hambre
    factor_peligro = np.clip(dist_pred/0.4, 0, 1)
    peso_n2 *= factor_peligro
    peso_n2 *= (1.0 + 0.3*L)
    peso_n2 = np.clip(peso_n2, 0, 0.7)

    # Peso de N+3: proporcional a la alerta con histeresis, no a la
    # distancia instantanea -- por eso sigue empujando la fuga incluso
    # unos frames despues de que el depredador salga del radio de
    # percepcion (esa es la histeresis operando).
    peso_n3 = np.clip(alerta_n3 * 0.65, 0, 0.65)
    suma_pesos = peso_n2 + peso_n3
    if suma_pesos > 1.0:
        factor_norm = 1.0 / suma_pesos
        peso_n2 *= factor_norm
        peso_n3 *= factor_norm

    motor_primario = (1.0-peso_n2-peso_n3)*motor_mot + peso_n2*motor_n2 + peso_n3*motor_n3
    motor = (1.0-border_stress)*motor_primario + border_stress*motor_nav
    if np.linalg.norm(motor) > 0:
        motor = motor/np.linalg.norm(motor)

    # --- 18. movimiento ---
    mag = np.linalg.norm(motor)
    if mag > 0:
        theta = np.arctan2(motor[1], motor[0])
        speed = (0.04 + 0.04*np.clip(mag, 0, 1)) * dt * (1.0 - reposo_intensidad)
        pos = pos + speed*motor
    pos = np.clip(pos, -0.95, 0.95)

    # ========================================================
    # 19. MAPA CORTICAL -- decaimiento exponencial de CMs activos
    # ========================================================
    # Difusion gaussiana local 3x3 sobre cada area 5x5
    def _blur5(v):
        p = np.pad(v, 1, mode='constant')
        k = np.array([[0.06, 0.12, 0.06],
                      [0.12, 0.28, 0.12],
                      [0.06, 0.12, 0.06]])
        out = np.zeros((5, 5))
        for i in range(5):
            for j in range(5):
                out[i, j] = np.sum(p[i:i+3, j:j+3] * k)
        return out

    cm = _blur5(activity_mot[:25].reshape(5, 5))
    cn = _blur5(activity_nav[:25].reshape(5, 5))
    c2 = _blur5(activity_n2[:25].reshape(5, 5))
    c3 = _blur5(activity_n3[:25].reshape(5, 5))

    # Factor de reposo continuo: 0 = lejos activo, 1 = en casa
    reposo = compuerta_en_casa

    # Decaimiento exponencial: mas rapido en reposo, mas lento cuando esta activa lejos
    decay_activo = 0.84
    decay_reposo = 0.70
    brain *= (decay_activo - (decay_activo - decay_reposo) * reposo)

    # Ganancia de inyeccion: alta cuando esta activa, baja en reposo
    gain = 0.50 * (1.0 - 0.45 * reposo)

    # Ruido espontaneo localizado (fresco cada frame, muy sutil)
    noise_amp = 0.03
    nm = np.random.rand(5, 5) * noise_amp
    nn = np.random.rand(5, 5) * noise_amp
    n2 = np.random.rand(5, 5) * noise_amp
    n3 = np.random.rand(5, 5) * noise_amp

    brain[7:12, 5:10] += cm * gain + nm
    brain[7:12, 12:17] += cn * gain + nn
    brain[13:18, 12:17] += c2 * gain + n2
    brain[13:18, 5:10] += c3 * gain * (0.3 + 0.7*alerta_n3) + n3

    # Bias de reposo: actividad minima sostenida cuando esta en casa (nunca desaparece)
    bias = 0.03 * reposo
    brain[7:12, 5:10] += bias
    brain[7:12, 12:17] += bias
    brain[13:18, 12:17] += bias
    brain[13:18, 5:10] += bias

    brain = np.clip(brain, 0, 0.80)

    # ========================================================
    # 19b. BRILLO HACIA EL AMARILLO cuando la criatura esta activa
    # ========================================================
    # nivel_actividad: 0 = en reposo total (en casa, poca hambre), 1 = activa/lejos
    nivel_actividad = 1.0 - reposo

    # Al reducir vmax cuando esta activa, los MISMOS valores de "brain"
    # quedan mas arriba en la escala de 'inferno' -> naranja/rojo se corre
    # hacia amarillo/blanco. No se toca brain (el dato "real"), solo el
    # mapeo de color -- es un efecto puramente visual de brillo.
    vmax_base = 0.80
    vmax_dinamico = vmax_base - 0.35 * nivel_actividad
    vmax_dinamico = np.clip(vmax_dinamico, 0.40, vmax_base)
    img.set_clim(vmin=0.0, vmax=vmax_dinamico)

    # --- 20. dibujo ---
    dot.set_data([pos[0]], [pos[1]])
    head = pos + 0.12*np.array([np.cos(theta), np.sin(theta)])
    line.set_data([pos[0], head[0]], [pos[1], head[1]])
    food_dot.set_data([food_pos[0]], [food_pos[1]])
    home_dot.set_data([home_pos[0]], [home_pos[1]])
    pred_dot.set_data([pred_pos[0]], [pred_pos[1]])
    if material_activo:
        material_patch.center = (material_pos[0], material_pos[1])
        material_patch.set_visible(True)
    else:
        material_patch.set_visible(False)
    img.set_data(brain)

    estado_nido = "COMPLETADO" if nido_completado else f"{int(np.floor(total_depositado))}/{materiales_maximos}"
    info_text.set_text(
        f"HAMBRE: {hunger:.2f}  SEGURIDAD: {safety:.2f}\n"
        f"IMPULSO: {impulso_construir:.2f}  URGENCIA: {urgencia_constructiva:.2f}\n"
        f"CARGA L: {L:.2f}  N+2 medio: {np.mean(activity_n2):.3f}\n"
        f"NIDO: {estado_nido}\n"
        f"EDAD: {edad:.2f}  INSTINTO: {instinto_construccion:.2f}\n"
        f"ALERTA N+3: {alerta_n3:.2f}  (tau={TAU_N3:.0f})"
    )
    return dot, line, img, food_dot, home_dot, pred_dot, material_patch, info_text

actualizar_nido()

try:
    ani = FuncAnimation(fig, update, interval=60, cache_frame_data=False)
    plt.tight_layout()
    plt.show(block=True)
except KeyboardInterrupt:
    pass
finally:
    sys.exit(0)