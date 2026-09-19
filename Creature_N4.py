# Cortical controlled Creature by: Oscar Chang   Ph.D.
# September 19, 2026

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
ANGLES = np.linspace(0, 2*np.pi, N, endpoint=False)
TIME_WARP = 1.0

# --- Softmax relaxation parameters (replace "decay + argmax") ---
# alpha: how fast activity relaxes toward the target each frame.
# temperature: how sharp the competition is (low temperature = almost
# a single winner; high temperature = multiple units share activity).
ALPHA_MOT, T_MOT = 0.35, 0.25
ALPHA_NAV, T_NAV = 0.40, 0.15
ALPHA_N2,  T_N2  = 0.35, 0.20
ALPHA_N4,  T_N4  = 0.35, 0.20

# --- N+4: threat hysteresis (danger memory) ---
# Instead of a "danger" that only grows with time (without real memory
# of the event), N+4 maintains an alert state that rises quickly when
# perceiving the predator nearby and decays SLOWLY with time constant TAU_N4.
# This prevents the creature from being surprised by the "ghost predator":
# even after the predator leaves the perception radius, alertness persists
# for ~TAU_N4 frames before shutting off.
PRED_PERCEPTION_RADIUS = 0.45   # radius within which the predator is "perceived"
TAU_N4 = 16.0                   # tau from the paper: alert decay time constant (frames)
ALERT_RISE_RATE = 0.9           # rise speed of alert when perceiving threat

# --- Maturation (replaces build_despierto) ---
AWAKE_AGE = 0.6
MATURATION_RHO = 12.0   # slope of the maturation sigmoid

# --- Continuous pickup (replaces tiene_brizna) ---
PICKUP_RADIUS = 0.12
NEST_RADIUS = 0.25
K_PICKUP = 0.55  # charge rate per frame when on top of material
K_DROP = 0.45    # discharge rate per frame when on top of nest
STEEPNESS_GATE = 50.0  # how abrupt the "I am on top" window is

MIN_IMPULSE = 0.15

def sigmoid_gate(x, edge=30.0):
    z = np.clip(-edge * x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))

def softmax_relax(activity, energies, alpha, temperature):
    """Continuous competition without argmax: activity relaxes toward a
    softmax distribution of energies. This is the standard mean-field
    approximation of the equilibrium of a lateral inhibition network type Amari
    (1977) -- it does not solve the full ODE step by step, but converges to
    the same type of equilibrium distribution (one region wins, the rest
    decays), without any step of the code asking "which is the maximum"."""
    e = energies - np.max(energies)
    weights = np.exp(e / temperature)
    target = weights / np.sum(weights)
    return (1.0 - alpha) * activity + alpha * target

# ============================================================
# WORLD STATE
# ============================================================

pos = np.array([0.0, 0.0])
theta = 0.0
hunger = 0.3
safety = 0.0
danger = 0.0
border_stress = 0.0
alert_n4 = 0.0   # hysteresis alert state (N+4)

food_pos = np.array([0.65, 0.45]); food_theta = 0.0; food_radius = 0.65
home_pos = np.array([-0.55, -0.35]); home_theta = np.pi; home_radius = 0.60
pred_pos = np.array([0.0, -0.7]); pred_theta = 0.0; pred_radius = 0.8

nest_pos = np.array([-0.70, 0.70])
nest_size = 0.30
nest_cells = 3
cell_size = nest_size / nest_cells

max_materials = 9
nest_completed = False
total_deposited = 0.0        # continuous integral of discharge flow
total_picked_fraction = 0.0  # continuous integral of pickup flow (to consume the object)

material_pos = np.array([0.0, 0.0])
material_active = False

# --- Replaces tiene_brizna: continuous load variable ---
L = 0.0

age = 0.0
build_instinct = 0.0

build_impulse = 0.8
base_impulse_rate = 0.015
build_urgency = 1.2

materials_spawned = 0
max_materials_spawned = 30
time_without_material = 0.0
time_without_building = 0.0

food_lock = False
home_lock = False

# ============================================================
# FOUR CORTICAL AREAS (same weight structure as before)
# ============================================================

activity_mot = np.ones(N) / N
cms_mot = []
for k in range(N):
    cms_mot.append({
        "angle": ANGLES[k],
        "food_weight": np.random.uniform(1.2, 1.6),
        "home_weight": np.random.uniform(1.3, 1.7),
        "pred_weight": np.random.uniform(1.2, 1.7),
        "explore": np.random.uniform(0, 0.06)
    })

activity_nav = np.ones(N) / N
cms_nav = []
for k in range(N):
    cms_nav.append({
        "angle": ANGLES[k],
        "border_weight": np.random.uniform(0.8, 1.2),
        "explore": np.random.uniform(0, 0.05)
    })

activity_n2 = np.ones(N) / N
cms_n2 = []
for k in range(N):
    cms_n2.append({
        "angle": ANGLES[k],
        "material_weight": np.random.uniform(1.2, 1.6),
        "nest_weight": np.random.uniform(1.3, 1.7),
        "explore": np.random.uniform(0, 0.04)
    })

# --- AREA N+4: escape with threat memory (hysteresis) ---
activity_n4 = np.ones(N) / N
cms_n4 = []
for k in range(N):
    cms_n4.append({
        "angle": ANGLES[k],
        "escape_weight": np.random.uniform(1.2, 1.7),
        "explore": np.random.uniform(0, 0.04)
    })

# ============================================================
# VISUAL CORTICAL MAP
# ============================================================

brain = np.zeros((21, 21))
# 2x2 compact layout, shifted down to avoid clashing with the variable
# text (which occupies the top strip, rows 0-6):
#   MOT (rows 7-11, left)   NAV (rows 7-11, right)
#   N+4 (rows 13-17, left)  N+2 (rows 13-17, right)
core_mot = [(r, c) for r in range(7, 12) for c in range(5, 10)]
core_nav = [(r, c) for r in range(7, 12) for c in range(12, 17)]
core_n4  = [(r, c) for r in range(13, 18) for c in range(5, 10)]
core_n2  = [(r, c) for r in range(13, 18) for c in range(12, 17)]

# ============================================================
# FIGURE
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.set_xlim(-1, 1); ax1.set_ylim(-1, 1)
ax1.set_facecolor("black")
ax1.set_title("WORLD - CONTINUOUS VERSION (no argmax, no boolean controls) + N+4 (threat hysteresis)", color='white', fontsize=9)

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

start_x = nest_pos[0] - nest_size / 2
start_y = nest_pos[1] - nest_size / 2
ax1.plot([start_x, start_x + nest_size], [start_y, start_y], color='yellow', linewidth=2, alpha=0.8)
ax1.plot([start_x, start_x + nest_size], [start_y + nest_size, start_y + nest_size], color='yellow', linewidth=2, alpha=0.8)
ax1.plot([start_x, start_x], [start_y, start_y + nest_size], color='yellow', linewidth=2, alpha=0.8)
ax1.plot([start_x + nest_size, start_x + nest_size], [start_y, start_y + nest_size], color='yellow', linewidth=2, alpha=0.8)
for i in range(1, nest_cells):
    x = start_x + i * cell_size
    ax1.plot([x, x], [start_y, start_y + nest_size], color='yellow', linewidth=1, alpha=0.5)
    y = start_y + i * cell_size
    ax1.plot([start_x, start_x + nest_size], [y, y], color='yellow', linewidth=1, alpha=0.5)

cells = []
for row in range(3):
    for col in range(3):
        x = start_x + col * cell_size
        y = start_y + row * cell_size
        cell = plt.Rectangle((x, y), cell_size, cell_size, facecolor='yellow', alpha=0.0, edgecolor='none')
        ax1.add_patch(cell)
        cells.append(cell)

img = ax2.imshow(brain, vmin=0, vmax=1, cmap='inferno')
ax2.axvline(10.5, color='white', linestyle=':', alpha=0.2, linewidth=0.5)
ax2.axhline(12, color='white', linestyle=':', alpha=0.2, linewidth=0.5)
ax2.text(7, 12.4, 'MOT', color='white', fontsize=8, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.7))
ax2.text(14, 12.4, 'NAV', color='white', fontsize=8, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='darkblue', alpha=0.7))
ax2.text(7, 18.4, 'N+4', color='white', fontsize=8, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='darkorange', alpha=0.7))
ax2.text(14, 18.4, 'N+3', color='white', fontsize=8, ha='center', va='top', bbox=dict(boxstyle='round', facecolor='darkgreen', alpha=0.7))
ax2.set_title("CORTEX (continuous softmax competition)", color='white', fontsize=9)
info_text = ax2.text(0.02, 0.98, "", transform=ax2.transAxes, color='white', fontsize=8,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# ============================================================
# AUXILIARY FUNCTIONS
# ============================================================

def update_nest():
    global cells
    filled_cells = min(int(np.floor(total_deposited)), max_materials)
    for cell in cells:
        cell.set_alpha(0.0)
    for i in range(filled_cells):
        cells[i].set_alpha(0.6)

def spawn_material():
    global material_pos, material_active, materials_spawned
    if materials_spawned >= max_materials_spawned or nest_completed:
        material_active = False
        return
    if material_active:
        return
    center_x, center_y, radius = 0.75, -0.75, 0.20
    for _ in range(200):
        angle = np.random.uniform(0, 2*np.pi)
        distance = np.random.uniform(0, radius)
        new_pos = np.array([center_x + distance*np.cos(angle), center_y + distance*np.sin(angle)])
        if np.linalg.norm(new_pos - nest_pos) > 0.20 and np.linalg.norm(new_pos - home_pos) > 0.20:
            material_pos = new_pos
            material_active = True
            materials_spawned += 1
            return
    material_active = False

def reset_system():
    global pos, theta, hunger, safety, danger, border_stress, alert_n4
    global activity_mot, activity_nav, activity_n2, activity_n4, brain
    global food_pos, home_pos, pred_pos, food_theta, home_theta, pred_theta
    global food_lock, home_lock
    global material_active, nest_completed
    global build_impulse, materials_spawned, time_without_material
    global time_without_building, build_urgency
    global age, build_instinct, L, total_deposited, total_picked_fraction

    try:
        winsound.Beep(180, 600)
    except Exception:
        pass

    pos = np.array([0.0, 0.0]); theta = 0.0
    hunger = 0.3; safety = 0.0; danger = 0.0; border_stress = 0.0; alert_n4 = 0.0
    activity_mot = np.ones(N)/N; activity_nav = np.ones(N)/N; activity_n2 = np.ones(N)/N; activity_n4 = np.ones(N)/N
    brain = np.zeros((21, 21))

    food_theta = np.random.uniform(0, 2*np.pi)
    home_theta = np.random.uniform(0, 2*np.pi)
    pred_theta = np.random.uniform(0, 2*np.pi)
    food_pos = np.array([food_radius*np.cos(food_theta), 0.55*np.sin(1.7*food_theta)])
    home_pos = np.array([home_radius*np.cos(home_theta), 0.45*np.sin(1.3*home_theta)])
    pred_pos = np.array([pred_radius*np.sin(pred_theta), pred_radius*np.cos(pred_theta)])

    food_lock = False; home_lock = False
    material_active = False; nest_completed = False
    L = 0.0; total_deposited = 0.0; total_picked_fraction = 0.0
    build_impulse = 0.8
    materials_spawned = 0; time_without_material = 0.0
    time_without_building = 0.0
    build_urgency = 1.2
    age = 0.0; build_instinct = 0.0
    update_nest()

# ============================================================
# UPDATE LOOP
# ============================================================

def update(frame):
    global pos, theta, hunger, safety, danger, border_stress, alert_n4
    global activity_mot, activity_nav, activity_n2, activity_n4, brain
    global food_pos, home_pos, pred_pos, food_theta, home_theta, pred_theta
    global food_lock, home_lock
    global material_active, material_pos, nest_completed
    global build_impulse, materials_spawned, time_without_material
    global time_without_building, build_urgency
    global age, build_instinct, L, total_deposited, total_picked_fraction

    dt = TIME_WARP

    # --- 1. distances ---
    dist_food = np.linalg.norm(pos - food_pos)
    dist_home = np.linalg.norm(pos - home_pos)
    dist_pred = np.linalg.norm(pos - pred_pos)
    dist_nest = np.linalg.norm(pos - nest_pos)
    dist_material = np.linalg.norm(pos - material_pos) if material_active else 999.0

    # --- 2. homeostasis ---
    hunger = np.clip(hunger + 0.0015*dt, 0, 1)
    safety = np.clip(safety + (0.0 if dist_home < 0.25 else 0.0020*dt) - (safety if dist_home < 0.25 else 0.0), 0, 1)
    if dist_home < 0.25:
        safety = 0.0
    danger = np.clip(danger + 0.003*dt, 0, 1)

    # --- continuous homeostatic gating (same as the working version) ---
    gate_at_home = sigmoid_gate(0.25 - dist_home)
    gate_low_hunger = sigmoid_gate(0.40 - hunger)
    gate_high_safety = sigmoid_gate(safety - 0.80)
    gate_urgency = sigmoid_gate(hunger - 0.50) * sigmoid_gate(0.40 - safety)

    attenuation = np.clip((1.0 - safety) / 0.20, 0, 1)
    hunger_attenuated = hunger * attenuation
    safety_attenuated = safety * 1.8
    hunger_urgent = hunger * 1.5
    safety_urgent = safety * 0.3
    hunger_flat = hunger
    safety_flat = safety

    weight_attenuated = gate_high_safety
    weight_urgent = gate_urgency * (1.0 - weight_attenuated)
    weight_flat = np.clip(1.0 - weight_attenuated - weight_urgent, 0, 1)

    effective_hunger_away = weight_attenuated*hunger_attenuated + weight_urgent*hunger_urgent + weight_flat*hunger_flat
    current_safety_need_away = weight_attenuated*safety_attenuated + weight_urgent*safety_urgent + weight_flat*safety_flat

    active_at_home = gate_at_home * (1.0 - gate_low_hunger)
    rest_intensity = gate_at_home * gate_low_hunger
    away = 1.0 - gate_at_home

    effective_hunger = active_at_home*hunger + away*effective_hunger_away
    current_safety_need = rest_intensity*1.0 + away*current_safety_need_away

    # --- 3. age and instinct: continuous sigmoid, no boolean flag ---
    age += 0.01 * dt
    build_instinct = sigmoid_gate(age - AWAKE_AGE, edge=MATURATION_RHO)

    # --- 4. constructive urgency (bookkeeping of progress, not motor control) ---
    if not nest_completed:
        build_urgency += 0.005*dt
        build_urgency += 0.015*dt * sigmoid_gate(dist_material - 999 + 1) * (1.0 - L)  # smooth contribution if material is visible and not being picked up
        build_urgency += 0.02*dt * L
        progress = total_deposited / max_materials
        build_urgency += 0.005*progress*dt
        time_without_building += dt
        build_urgency += 0.01*dt * sigmoid_gate(time_without_building - 60, edge=0.3)
        build_urgency = np.clip(build_urgency, 0, 1.5)

    # --- 5. border stress ---
    dist_to_wall_x = 1.0 - abs(pos[0])
    dist_to_wall_y = 1.0 - abs(pos[1])
    closest_wall_dist = min(dist_to_wall_x, dist_to_wall_y)
    border_stress = (np.clip((0.25 - closest_wall_dist)/0.25, 0, 1) ** 2) if closest_wall_dist < 0.25 else 0.0

    # --- 6. entity movement ---
    food_theta += 0.015*dt
    food_pos = np.array([food_radius*np.cos(food_theta), 0.55*np.sin(1.7*food_theta)])
    home_theta -= 0.010*dt
    home_pos = np.array([home_radius*np.cos(home_theta), 0.45*np.sin(1.3*home_theta)])
    home_zone.set_center((home_pos[0], home_pos[1]))
    home_safety_zone.set_center((home_pos[0], home_pos[1]))

    # --- 7. predator ---
    pred_speed = 0.023*dt
    if dist_home < 0.30:
        pred_theta += np.random.uniform(-0.8, 0.8)
    else:
        to_prey = pos - pred_pos
        pred_theta = np.arctan2(to_prey[1], to_prey[0])
    pred_pos = pred_pos + pred_speed*np.array([np.cos(pred_theta), np.sin(pred_theta)])
    pred_pos = np.clip(pred_pos, -1.2, 1.2)

    # ========================================================
    # 7b. N+4: PERCEPTION AND ALERT WITH HYSTERESIS
    # ========================================================
    # threat_perception: smooth window (sigmoid) of "the predator is
    # inside the perception radius", not a hard threshold boolean.
    threat_perception = sigmoid_gate(PRED_PERCEPTION_RADIUS - dist_pred, edge=25.0)

    # Leaky integrator with explicit temporal asymmetry:
    #   - rise: fast, proportional to current perception (immediate reaction)
    #   - fall: slow, with time constant TAU_N4 (memory of danger)
    # This is exactly what the baseline without N+4 lacked: there "danger"
    # only grew with time and never really responded to the event of
    # perceiving the predator, so the creature had no way to sustain
    # caution after the predator disappeared from view (the "ghost
    # predator" caught it systematically).
    alert_rise = ALERT_RISE_RATE * threat_perception * (1.0 - alert_n4)
    alert_decay = (alert_n4 / TAU_N4) * (1.0 - threat_perception)
    alert_n4 = np.clip(alert_n4 + dt*(alert_rise - alert_decay), 0, 1)

    # --- 8. collisions (objective world events, not brain events) ---
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

    # --- 9. constructive impulse (continuous bookkeeping, no artificial floor) ---
    if nest_completed:
        build_impulse *= 0.95
    else:
        build_impulse += base_impulse_rate*dt
        build_impulse += build_urgency*0.02*dt
        build_impulse += 0.01*dt * (1.0 - L) * (1.0 if material_active else 0.0)
        build_impulse += 0.02*dt * L
        progress = total_deposited / max_materials
        build_impulse += 0.005*progress*dt
        build_impulse += 0.01*dt * sigmoid_gate(time_without_building - 50, edge=0.3)
    build_impulse = np.clip(build_impulse, 0, 1)

    # ========================================================
    # 10. CONTINUOUS LOAD L(t) -- replaces tiene_brizna
    # ========================================================
    # g_pickup/g_drop: "I am on top of material" / "I am on top of nest",
    # as smooth windows (sigmoids), not as isolated boolean comparisons.
    g_pickup = sigmoid_gate(PICKUP_RADIUS - dist_material, edge=STEEPNESS_GATE) if material_active else 0.0
    g_drop = sigmoid_gate(NEST_RADIUS - dist_nest, edge=STEEPNESS_GATE) if not nest_completed else 0.0

    flow_in = K_PICKUP * g_pickup * (1.0 - L)
    flow_out = K_DROP * g_drop * L
    L = np.clip(L + dt*(flow_in - flow_out), 0, 1)

    # The count of "how much has been delivered" is the integral of the SAME
    # flow that empties L -- not a separate threshold detection.
    total_deposited = min(max_materials, total_deposited + dt*flow_out)
    total_picked_fraction += dt*flow_in
    if total_picked_fraction >= 1.0 and material_active:
        # finished "loading" one complete unit of material: the discrete
        # world object is consumed (environment event, not brain event)
        material_active = False
        total_picked_fraction -= 1.0
        try: winsound.Beep(800, 100)
        except Exception: pass

    if total_deposited >= max_materials:
        nest_completed = True
        material_active = False
    update_nest()

    # --- 11. generate new material ---
    if not nest_completed and not material_active:
        if materials_spawned < max_materials_spawned:
            spawn_material()
        else:
            time_without_material += dt
            if time_without_material > 150:
                materials_spawned = 0
                time_without_material = 0.0
                spawn_material()

    # --- 12. sensory directions ---
    vec_food = food_pos - pos; vec_home = home_pos - pos; vec_pred = pred_pos - pos
    angle_food = np.arctan2(vec_food[1], vec_food[0])
    angle_home = np.arctan2(vec_home[1], vec_home[0])
    angle_pred = np.arctan2(vec_pred[1], vec_pred[0])

    force_west = 1.0/(1.0 + (pos[0]-(-0.95)))
    force_east = 1.0/(1.0 + (0.95-pos[0]))
    force_south = 1.0/(1.0 + (pos[1]-(-0.95)))
    force_north = 1.0/(1.0 + (0.95-pos[1]))
    vec_border = np.array([force_east-force_west, force_north-force_south])
    angle_border = np.arctan2(vec_border[1], vec_border[0])

    # ========================================================
    # 13. AREA 1 (MOT): energies -> softmax relaxation (no argmax)
    # ========================================================
    # N+4 active: the MOT avoidance term no longer uses "danger"
    # which only grew with time -- it uses the hysteresis alert, which
    # rises when perceiving the predator and persists ~TAU_N4 frames after.
    current_danger_factor = 0.0 if dist_home < 0.25 else alert_n4
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
    activity_mot = softmax_relax(activity_mot, energies_mot, ALPHA_MOT, T_MOT)

    # ========================================================
    # 14. AREA 2 (NAV): same, with rest flicker continuously mixed in
    # ========================================================
    angs_n = np.array([cm["angle"] for cm in cms_nav])
    bw = np.array([cm["border_weight"] for cm in cms_nav])
    exn = np.array([cm["explore"] for cm in cms_nav])
    energies_nav = -border_stress*bw*np.cos(angle_border-angs_n) + exn*np.random.uniform(-1, 1, N)
    rest_noise = np.random.uniform(-1, 1, N) * 2.0
    energies_nav_eff = (1.0 - rest_intensity)*energies_nav + rest_intensity*rest_noise
    activity_nav = softmax_relax(activity_nav, energies_nav_eff, ALPHA_NAV, T_NAV)

    # ========================================================
    # 15. AREA 3 (N+2): formula was already written as interpolation
    # -- now L is truly continuous, not 0/1
    # ========================================================
    vec_material = (material_pos - pos) if material_active else np.array([0.0, 0.0])
    vec_nest = nest_pos - pos
    angle_material = np.arctan2(vec_material[1], vec_material[0]) if np.linalg.norm(vec_material) > 0 else 0.0
    angle_nest = np.arctan2(vec_nest[1], vec_nest[0])
    material_available = 1.0 if material_active else 0.0

    angs2 = np.array([cm["angle"] for cm in cms_n2])
    mw = np.array([cm["material_weight"] for cm in cms_n2])
    nw = np.array([cm["nest_weight"] for cm in cms_n2])
    ex2 = np.array([cm["explore"] for cm in cms_n2])
    material_align = np.cos(angle_material-angs2) if material_active else np.zeros(N)
    nest_align = np.cos(angle_nest-angs2)

    energies_n2 = (
        (1-L)*material_available*mw*material_align*(0.8+0.5*build_instinct)
        + L*nw*nest_align*(0.7+0.5*build_instinct)
        + L*mw*material_align*0.1
        + (1-L)*material_available*nw*nest_align*0.1
        + (1-L)*(1-material_available)*ex2*np.random.uniform(0.5, 1.5, N)*(0.3+build_instinct)
        + ex2*np.random.uniform(-1, 1, N)
    )
    activity_n2 = softmax_relax(activity_n2, energies_n2, ALPHA_N2, T_N2)

    # ========================================================
    # 15b. AREA 4 (N+4): escape energies, active only while
    # alert_n4 > 0 -- the hysteresis alert itself decides how long
    # this area stays "on" after losing sight of the predator.
    # ========================================================
    angle_escape = angle_pred + np.pi  # direction opposite to predator
    angs4 = np.array([cm["angle"] for cm in cms_n4])
    ew4 = np.array([cm["escape_weight"] for cm in cms_n4])
    ex4 = np.array([cm["explore"] for cm in cms_n4])
    escape_align4 = np.cos(angle_escape - angs4)

    energies_n4 = (
        alert_n4 * ew4 * escape_align4
        + ex4 * np.random.uniform(-1, 1, N)
        + 0.06 * np.random.randn(N)
    )
    activity_n4 = softmax_relax(activity_n4, energies_n4, ALPHA_N4, T_N4)

    # ========================================================
    # 16. MOTOR SYNTHESIS (population-level, same structure as always)
    # ========================================================
    vecs = np.stack([np.cos(ANGLES), np.sin(ANGLES)], axis=1)
    motor_mot = activity_mot @ vecs
    motor_nav = activity_nav @ vecs
    motor_n2 = activity_n2 @ vecs
    motor_n4 = activity_n4 @ vecs
    if np.linalg.norm(motor_mot) > 0: motor_mot = motor_mot/np.linalg.norm(motor_mot)
    if np.linalg.norm(motor_nav) > 0: motor_nav = motor_nav/np.linalg.norm(motor_nav)
    if np.linalg.norm(motor_n2) > 0: motor_n2 = motor_n2/np.linalg.norm(motor_n2)
    if np.linalg.norm(motor_n4) > 0: motor_n4 = motor_n4/np.linalg.norm(motor_n4)

    # ========================================================
    # 17. INTEGRATION -- all continuous, no "if build_despierto"
    # ========================================================
    g_impulse = sigmoid_gate(build_impulse - MIN_IMPULSE, edge=40.0)
    weight_n2 = build_instinct * g_impulse * build_impulse * 0.7
    hunger_factor = 1.0 - hunger*0.5
    weight_n2 *= hunger_factor
    danger_factor = np.clip(dist_pred/0.4, 0, 1)
    weight_n2 *= danger_factor
    weight_n2 *= (1.0 + 0.3*L)
    weight_n2 = np.clip(weight_n2, 0, 0.7)

    # Weight of N+4: proportional to the hysteresis alert, not the
    # instantaneous distance -- that's why it keeps pushing escape even
    # a few frames after the predator leaves the perception radius (that's
    # the hysteresis in action).
    weight_n4 = np.clip(alert_n4 * 0.65, 0, 0.65)
    sum_weights = weight_n2 + weight_n4
    if sum_weights > 1.0:
        norm_factor = 1.0 / sum_weights
        weight_n2 *= norm_factor
        weight_n4 *= norm_factor

    motor_primary = (1.0-weight_n2-weight_n4)*motor_mot + weight_n2*motor_n2 + weight_n4*motor_n4
    motor = (1.0-border_stress)*motor_primary + border_stress*motor_nav
    if np.linalg.norm(motor) > 0:
        motor = motor/np.linalg.norm(motor)

    # --- 18. movement ---
    mag = np.linalg.norm(motor)
    if mag > 0:
        theta = np.arctan2(motor[1], motor[0])
        speed = (0.04 + 0.04*np.clip(mag, 0, 1)) * dt * (1.0 - rest_intensity)
        pos = pos + speed*motor
    pos = np.clip(pos, -0.95, 0.95)

    # ========================================================
    # 19. CORTICAL MAP -- exponential decay of active CMs
    # ========================================================
    # Local 3x3 gaussian diffusion over each 5x5 area
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
    c4 = _blur5(activity_n4[:25].reshape(5, 5))

    # Rest factor: 0 = active/away, 1 = at home resting
    rest = gate_at_home

    # Exponential decay: faster when resting, slower when active away
    decay_active = 0.84
    decay_rest = 0.70
    brain *= (decay_active - (decay_active - decay_rest) * rest)

    # Injection gain: high when active, low when resting
    gain = 0.50 * (1.0 - 0.45 * rest)

    # Spontaneous localized noise (fresh each frame, very subtle)
    noise_amp = 0.03
    nm = np.random.rand(5, 5) * noise_amp
    nn = np.random.rand(5, 5) * noise_amp
    n2 = np.random.rand(5, 5) * noise_amp
    n4 = np.random.rand(5, 5) * noise_amp

    brain[7:12, 5:10] += cm * gain + nm
    brain[7:12, 12:17] += cn * gain + nn
    brain[13:18, 12:17] += c2 * gain + n2
    brain[13:18, 5:10] += c4 * gain * (0.3 + 0.7*alert_n4) + n4

    # Rest bias: minimum sustained activity when at home (never disappears)
    bias = 0.03 * rest
    brain[7:12, 5:10] += bias
    brain[7:12, 12:17] += bias
    brain[13:18, 12:17] += bias
    brain[13:18, 5:10] += bias

    brain = np.clip(brain, 0, 0.80)

    # ========================================================
    # 19b. BRIGHTNESS TOWARD YELLOW when the creature is active
    # ========================================================
    # activity_level: 0 = fully resting (at home, little hunger), 1 = active/away
    activity_level = 1.0 - rest

    # By reducing vmax when active, the SAME "brain" values are pushed
    # higher on the 'inferno' scale -> orange/red shifts toward
    # yellow/white. The brain data itself is untouched -- it's a purely
    # visual brightness effect.
    vmax_base = 0.80
    vmax_dynamic = vmax_base - 0.35 * activity_level
    vmax_dynamic = np.clip(vmax_dynamic, 0.40, vmax_base)
    img.set_clim(vmin=0.0, vmax=vmax_dynamic)

    # --- 20. drawing ---
    dot.set_data([pos[0]], [pos[1]])
    head = pos + 0.12*np.array([np.cos(theta), np.sin(theta)])
    line.set_data([pos[0], head[0]], [pos[1], head[1]])
    food_dot.set_data([food_pos[0]], [food_pos[1]])
    home_dot.set_data([home_pos[0]], [home_pos[1]])
    pred_dot.set_data([pred_pos[0]], [pred_pos[1]])
    if material_active:
        material_patch.center = (material_pos[0], material_pos[1])
        material_patch.set_visible(True)
    else:
        material_patch.set_visible(False)
    img.set_data(brain)

    nest_status = "COMPLETED" if nest_completed else f"{int(np.floor(total_deposited))}/{max_materials}"
    info_text.set_text(
        f"HUNGER: {hunger:.2f}  SAFETY: {safety:.2f}\n"
        f"IMPULSE: {build_impulse:.2f}  URGENCY: {build_urgency:.2f}\n"
        f"LOAD L: {L:.2f}  N+2 mean: {np.mean(activity_n2):.3f}\n"
        f"NEST: {nest_status}\n"
        f"AGE: {age:.2f}  INSTINCT: {build_instinct:.2f}\n"
        f"ALERT N+4: {alert_n4:.2f}  (tau={TAU_N4:.0f})"
    )
    return dot, line, img, food_dot, home_dot, pred_dot, material_patch, info_text

update_nest()

try:
    ani = FuncAnimation(fig, update, interval=60, cache_frame_data=False)
    plt.tight_layout()
    plt.show(block=True)
except KeyboardInterrupt:
    pass
finally:
    sys.exit(0)
