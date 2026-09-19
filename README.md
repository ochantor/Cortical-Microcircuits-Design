▶️ **Watch the complete demonstration:** https://youtu.be/r9juAT-AcI4
<video src="Real%20Creature%20Movie.mp4" controls autoplay loop muted width="100%">
</video>

# Cortical-Microcircuits-Design
How Canonical Cortical Microcircuits Give Rise to Intelligent Behavior Without Learning


> *"Intelligence is not learning. It is the awakening of tissue that evolution has been saving for you."*
>
> *"If nature only has CMs at its disposal to build the cortex, then everything we call learning, memory, or thought is nothing more than the modulation of the activity of those very same CMs in space-time."*

---
The most advanced biological intelligence on Earth does not look like modern artificial intelligence. It is not built from feed-forward layer upon layer of deep, gigantic, monolithic neural connections.

Instead, the mammalian neocortex is an intricate, vast landscape formed by hundreds of thousands—millions—of microscopic, repeating cells that tightly cover the entire cortical surface like a vast, biological honeycomb. These are the cortical microcircuits (CM), or cortical minicolumns. First discovered and conceptualized by the pioneering neurophysiologist Vernon Mountcastle in 1957, these canonical structures represent the fundamental, modular computational units of mammalian cognition, including humans.

Artificial systems typically treat the brain as a massive, deep feed-forward processor that transforms inputs into explicit behavioral representations. This repository challenges that assumption by looking directly at the raw, localized dynamics of the cortical honeycomb.
A new AI concept built directly with mammalian cortical elements—microcircuits, cortical areas, lateral competition, temporal amygdalar asymmetry—demonstrates that a single new cortical area with hysteresis resolves long-horizon survival, revealing that the embodied bottleneck is temporal, not representational

![Cortical Microcircuits](Cortical%20Microcircuits.png)

## The Experiment

Run the program. You will see a small white creature in a dark world.

At first, it only survives. It eats when hungry, flees from the predator, and takes refuge when danger lurks. This is **N+1** (motor) and **N+2** (navigation): the first operational cortical tissues, assembled from rotated microcircuits that burn energy and produce survival behavior.

Then, at age `0.6`, something changes. A third tissue, **N+3** (sensorimotor / nest-building), awakens. The creature that only knew how to survive now builds a nest—searching, collecting, carrying, and depositing materials until a 9-cell structure is complete. No learning. No synaptic modification. Only tissue that was preconfigured, waiting for its moment.

**But something else is happening, and it is the most important part of the experiment.**

Watch the predator. When it enters the creature's perceptual radius, the creature flees. That is expected. But now **watch what happens when the predator leaves the field of view.** The creature does not immediately return to foraging. It maintains evasive behavior. It acts *as if* the danger were still present. For approximately `TAU_N4` frames, a hidden variable—an internal state of alert—continues to bias the motor competition away from the predator's last known quadrant.

This is **N+4** (threat hysteresis): a cortical tissue found by expert human and LLM co-work. It does not merely process the present. It sustains a **state of mind**.

---

## The N-Series as Tissue Awakening

| Level | State | Emergent Function |
|:-----:|-------|-------------------|
| **N** | Isolated CMs, unassembled, undeployed | Pure potential, no behavior |
| **N+1** | CMs assembled and rotated in physical space (MOT) | Eat, flee, take refuge |
| **N+2** | CMs rotated in navigation space (NAV) | Border avoidance, spatial constraints |
| **N+3** | New tissue awakens. CMs rotated in construction space | Build nest |
| **N+4** | **New tissue awakens. CMs rotated in threat-memory space.** | **Sustain internal alert; decouple behavior from immediate perception** |
| N+5 | *(theoretical)* CMs rotated in planning space | Sequence planning |
| N+6 | *(theoretical)* CMs rotated in social space | Theory of mind |
| N+7 | *(theoretical)* CMs rotated in reflective space | Emergent consciousness |
| **N+∞** | Total integration of all tissues | Unity of the system |

Each level is a new tissue that awakens, not learning that accumulates.

---

## N+4: The Threat-Memory Tissue (Why This Changes Everything)

N+4 is not a "fear reflex." It is a **self-sustaining cortical field** that introduces *temporal depth* into the creature's behavior.

### The Mechanism

Like the other tissues, N+4 is a canonical tissue of 25 cortical microcircuits (CMs), systematically rotated across the directional manifold. But its energy landscape is governed by a hidden state variable, `alert_n4`, with asymmetric dynamics:

```python
# Perception is soft, not boolean
threat_perception = sigmoid(PRED_PERCEPTION_RADIUS - dist_pred)

# Hysteresis: fast rise, slow decay
rise = ALERT_RISE_RATE * threat_perception * (1.0 - alert_n4)
decay = (alert_n4 / TAU_N4) * (1.0 - threat_perception)
alert_n4 += dt * (rise - decay)
```

- **Rise time**: Immediate. When the predator is perceived, the tissue ignites.
- **Decay time**: Slow (`TAU_N4 = 16` frames). When the predator disappears, the tissue does not shut off. It burns energy at a diminishing rate, continuing to inject competitive bias into the motor hierarchy.

### The State of Mind

N+4 creates an **attractor state** in the creature's cortical dynamics. While `alert_n4 > 0`:

1. The N+4 tissue competes for motor control via `weight_n4 = alert_n4 * 0.65`.
2. The energy vector of N+4 points *away* from the predator (`angle_escape = angle_pred + π`).
3. The creature's motor output is a weighted mixture: `motor = (1-weight_n4)*motor_primary + weight_n4*motor_n4`.

The result is behavior that is **no longer a function of current sensory input alone**. It is a function of *history*, encoded as a persistent pattern of neural activity. The creature has, for the first time, an **internal model of the world that outlives its immediate perception**.

This is the difference between a reflex and a state of mind.

### Why This Matters for the Theory

Before N+4, the creature was a sophisticated stimulus-response machine. Its behaviors were complex but *reactive*: the motor vector was always a function of the current sensory field.

N+4 breaks this coupling. It demonstrates that the rotated canonical tissue architecture can support **persistent internal states**—the prerequisite for:

- **Working memory** (holding information across time)
- **Temporal credit assignment** (linking past events to present actions)
- **Anticipatory behavior** (acting on expected, not just observed, states)
- **Emotion** (a sustained internal tone that colors all other processing)

N+4 is the first tissue that does not just tell the creature *where* to go. It tells the creature *how to feel about the world* while it decides.

---

## The Rotated Canonical Tissue: Architecture

The entire system is built from one fundamental brick:

1. **The Cortical Microcircuit (CM)**: A computational unit with homeostatic weights (food, safety, threat, material, nest, escape).
2. **Rotation**: 25 copies of the CM are deployed, each tuned to a preferred angle, covering the full directional circle.
3. **Competition**: At each timestep, all CMs in a tissue receive sensory information, compute directional alignment, and compete via softmax relaxation. Activities decay exponentially.
4. **Integration**: Tissues compete hierarchically for motor control. MOT (survival), NAV (navigation), N+3 (construction), and N+4 (threat memory) are mixed continuously via sigmoid-gated weights.
5. **Awakening**: New tissues are activated by maturation signals (`build_instinct` for N+3, `alert_n4` for N+4), not by learning.

N is potential. N+1 is survival. N+2 is navigation. N+3 is construction. **N+4 is the birth of an inner life.**

---

## Why No Learning?

The creature moves from pure survival to nest-building to threat-mediated evasion **without a single synaptic weight changing**. All CM parameters are fixed at initialization. There is no Hebbian plasticity, no reinforcement, no backpropagation.

Behavior emerges because:

- New tissues awaken at the right developmental moment.
- These tissues are preconfigured by phylogeny (evolution) to encode useful behavioral manifolds.
- Competition and decay among rotated CMs transform static weights into dynamic, adaptive trajectories.

| Aspect | Tissue Awakening (Phylogeny) | Synaptic Plasticity (Ontogeny) |
|--------|------------------------------|-------------------------------|
| Energy Cost | Low (activate existing tissue) | High (modify synaptic structure) |
| Latency | Instantaneous upon awakening | Slow (requires trial and error) |
| Reliability | High (evolution-tested) | Low (can learn suboptimal strategies) |
| Heritability | Fully heritable | Non-heritable |
| Scalability | Additive (deploy new tissue) | Combinatorial (local rewiring) |

Synaptic plasticity is not the foundation of intelligence. It is a repair mechanism—a fallback for cases where evolution could not preconfigure the correct tissue. This program demonstrates the primacy of **phylogenetic preconfiguration** over **ontogenetic acquisition**.

---

## Philosophical Implications

### 1. Intelligence is the decompression of a phylogenetic archive
The genome does not encode behaviors. It encodes *tissues*. At the right age, each tissue awakens and decompresses a new behavioral manifold. The creature does not learn to build; the N+3 tissue awakens. The creature does not learn to remember danger; the N+4 tissue is put to work by human-LLM co-work.

### 2. The first "state of mind" requires no cortex—only persistence
N+4 shows that a "state of mind"—a sustained internal condition that modulates all behavior—can emerge from a single layer of recurrent competition with slow decay. You do not need a prefrontal cortex to have memory. You only need tissue that burns energy longer than the stimulus lasts.

### 3. Consciousness is the integration of all awakened tissues
N+1 gives you a body. N+3 gives you a project. **N+4 gives you a biography**—a creature that acts not just on what it sees, but on what it has *lived through*. When all tissues are awake and integrated, the system does not merely process information. It *has an experience of processing information*.

---

## Running the Experiment

```bash
python Creature_N4_OK.py
```

### Requirements
- Python 3.x
- NumPy
- Matplotlib (TkAgg backend)

### What to Observe
1. **0.0–0.6 age**: The creature survives using only MOT and NAV. Watch the cortical map (right panel): only MOT (red) and NAV (blue) are active.
2. **Age ~0.6**: N+3 (green) ignites. The creature begins seeking yellow materials and carrying them to the nest.
3. **When the predator approaches**: N+4 (orange) erupts in the cortical map. The creature flees.
4. **After the predator retreats**: Keep watching the orange cluster. It persists. The creature continues to avoid the predator's quadrant even when the green dot is far away. The `ALERT N+4` readout decays slowly. This is a **state of mind**, visible in real time.

---

## Figure 3: Ablation over the Hysteresis Time Constant

The paper's central figure shows the qualitative transition:

![Figure 3](Figure3_ablation.png)

- **x-axis**: `TAU_N4` (0–32 frames)
- **y-axis**: nest completion rate (%)
- **Result**: sigmoidal transition with critical threshold `τ_c ≈ 7.1 frames`; saturation at 100% for `TAU_N4 ≳ 12`

Each point is 30 independent trials. The logistic fit yields `R² = 0.997`.

To reproduce: modify `TAU_N4` in the code and run 30 trials per condition.

---

## Nomenclature Note (Code ↔ Paper)

| Paper | Code | Role |
|-------|------|------|
| N+1 | `MOT` | Motor |
| N+2 | `NAV` | Navigation |
| N+3 | `N+2` (historical identifier) | Nest-building |
| **N+4** | **`N+4`** | **Threat hysteresis** |

The nest-building area retains its historical identifier `N+2` in the code, corresponding to area `N+3` in the manuscript. The threat-hysteresis area, discovered via human-LLM co-design, is consistently named `N+4` in both code and manuscript.

---

## References

- Mountcastle, V.B. (1957). Modality and topographic properties of single neurons of cat's somatic sensory cortex.
- Tononi, G. (2004). An information integration theory of consciousness.
- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Amari, S. (1977). Dynamics of pattern formation in lateral-inhibition type neural fields.
- Chang, O. (2010). Evolving Cooperative Neural Agents for Controlling a Vision Guided Mobile Robot. *IEEE UKRICIS*. DOI: 10.1109/UKRICIS.2010.5898127.
- Chang, O. (2025). Cortical Microcircuits: The Functional Benchmark for AGI.
- Chang, O., & Perez, J. (2026). Threat Hysteresis as a Minimal Mechanism for Long-Horizon Survival in an Embodied Cortical Agent. (Manuscript submitted to Frontiers.)

---

## Contact & Collaboration

This is an active research program. We are seeking collaborators in:

- **Computational Neuroscience**: Modeling persistent activity and attractor dynamics in canonical cortical circuits.
- **Developmental Biology**: Mapping tissue awakening timelines to critical periods in biological brains.
- **Artificial Intelligence**: Extending the N-series to N+5 (planning), N+6 (social inference), and beyond—without learning algorithms.
- **Philosophy of Mind**: Investigating whether N+4 constitutes the simplest form of "intentionality" or "inner experience."

If you are working on persistent neural states, critical period plasticity, or alternatives to gradient-based learning in AI, we want to talk to you.

**Open an issue, start a discussion, or reach out.**

> *"The mind is not in the brain. The mind is the brain rotated in space-time."*

---

## Citation

If you use this code or the N-series framework in your research, please cite:

```bibtex
@article{chang2026threat,
  title={Threat Hysteresis as a Minimal Mechanism for Long-Horizon Survival in an Embodied Cortical Agent},
  author={Chang, Oscar and Perez, Jonathan},
  journal={Frontiers},
  year={2026},
  note={Manuscript submitted}
}
```

---

*License: GPL-3.0*
```

---

## Archivos auxiliares recomendados (para subir junto al README)

### `requirements.txt`
```
numpy
matplotlib
```

### `LICENSE`
Copia el texto completo de GPL-3.0 desde https://www.gnu.org/licenses/gpl-3.0.txt y guárdalo como `LICENSE` en la raíz del repo.

### `CITATION.cff` (opcional pero recomendado)
```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
authors:
  - family-names: Chang
    given-names: Oscar
  - family-names: Perez
    given-names: Jonathan
title: "Cortical-Microcircuits-Design: Threat Hysteresis as a Minimal Mechanism for Long-Horizon Survival"
version: 2.0
date-released: 2026-01-01
url: "https://github.com/tu-usuario/Cortical-Microcircuits-Design"
license: GPL-3.0
```

---

## Estructura final del repositorio

```
Cortical-Microcircuits-Design/
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── Creature_N4_OK.py
├── Figure3_ablation.png
├── Figure3_ablation.pdf
├── Cortical Microcircuits.png
└── Real Creature Movie.mp4
```

---

