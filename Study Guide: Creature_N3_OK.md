 # The N+3 Creature: A Student's Field Guide

**Run it. Watch a mind ignite.**

A white dot in a dark world. Four cortical tissues. Zero learning. One internal state of mind.

---

## What You See

| Symbol | Object | What the creature does |
|--------|--------|------------------------|
| `★` Red | Food | Eats when hungry |
| `○` Blue | Home | Rests, feels safe |
| `○` Green | Predator | **Flees — and keeps fleeing even after it vanishes** |
| `□` Yellow | Nest | Builds it, cell by cell, after maturation |
| `⬭` Yellow | Material | Gathers, carries, deposits |

**The right panel** is its living cortex. Four colored blobs competing for control in real time.

> **No `if` statements drive behavior. Only competition, decay, and hysteresis.**

---

## The 3 Laws (in 3 Functions)

### 1. `_compuerta(x)` — The Soft Switch
A sigmoid, not a boolean. Returns `0.0` to `1.0`. The creature never asks "am I home?" It asks "how home am I?"

```python
compuerta_en_casa = _compuerta(0.25 - dist_home)  # ~1.0 when close, ~0.0 when far
```

### 2. `relajar_softmax(act, energy, α, T)` — Neural Competition
25 directional units fight. No `argmax`. No single winner. The activity vector **relaxes** toward a softmax of energies:

```python
return (1-α)*actividad + α*softmax(energia / T)
```

- `α` = inertia (how fast it switches attention)
- `T` = temperature (low = dictatorship, high = democracy)

### 3. `alerta_n3` — The State of Mind
Fast rise. Slow decay. This is **working memory made of hysteresis:**

```python
subida = 0.9 * threat_seen * (1 - alerta_n3)
bajada = (alerta_n3 / 16.0) * (1 - threat_seen)
alerta_n3 += subida - bajada
```

The predator leaves. The fear stays. **That is N+3.**

---

## Anatomy of a Single Frame

```
Sense distances → Update homeostasis → Age the creature
      ↓
Compute 4 energy vectors (MOT, NAV, N+2, N+3)
      ↓
Relax each via softmax → Get 4 activity maps
      ↓
Blend into one motor vector: (1-w₂-w₃)·MOT + w₂·N+2 + w₃·N+3
      ↓
Move → Inject into cortical map → Decay → Diffuse → Render
```

**MOT** (red): hungry? → food. scared? → flee. tired? → home.  
**NAV** (blue): borders are lava. At home, injects noise = rest.  
**N+2** (green): awakens at age `0.6`. Material → Nest. No instructions.  
**N+3** (orange): **ignites on threat. Persists after threat. Biases all decisions.**

---

## The Core Trick

N+3 does not tell the creature *where* the predator is. It tells the creature *how afraid to be*. That fear is a **multiplicative weight** on the motor blend:

```python
peso_n3 = np.clip(alerta_n3 * 0.65, 0, 0.65)
```

Even when the predator is invisible, `alerta_n3 > 0` keeps pushing the creature away from the last known danger quadrant. **Behavior decoupled from sensation.**

---

## 5 Experiments to Break It

| Change | What to watch | Concept |
|--------|---------------|---------|
| `TAU_N3 = 100.0` | Fear never dies | Time constant of internal state |
| `ALPHA_MOT = 0.05` | Sluggish, drunk creature | Neural inertia |
| `EDAD_DESPIERTE = 0.1` | Builds immediately | Maturation as switch |
| Delete `+ peso_n3 * motor_n3` | No post-trauma avoidance | Functional lesion |
| `K_PICKUP = 0.05` | Must sit on material forever | First-order load dynamics |

---

## The One Question

> The creature flees 10 frames after the predator disappears. Where is the memory?

- a) The predator's coordinates  
- b) **`alerta_n3`** ← **Correct. A self-sustaining state variable. No synapses changed.**  
- c) Modified weights  
- d) Position history

**Intelligence here is not stored. It is sustained by four active cortical areas with 25 CMs each.**

---

**Run it. Find the orange glow that outlives the green dot. That is a mind.**

*Code: `Creature_N3_OK.py`*
