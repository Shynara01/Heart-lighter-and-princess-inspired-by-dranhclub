# heart
import numpy as np
import pygame

pygame.init()

# window
W, H = 800, 800
CENTER = np.array([W // 2, H // 2], dtype=int)
screen = pygame.display.set_mode((W, H), pygame.SRCALPHA)
pygame.display.set_caption("Heart Beat — NumPy only")
clock = pygame.time.Clock()

# heart curve
def heart_xy(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Parametric heart for array t (radians).
    """
    x = 16.0 * np.sin(t) ** 3
    y = (
        13.0 * np.cos(t)
        - 5.0 * np.cos(2.0 * t)
        - 2.0 * np.cos(3.0 * t)
        - 1.0 * np.cos(4.0 * t)
    )
    return x, y

# Samples around 0..2π
SAMPLES = 2400
TS = np.linspace(0.0, 2.0 * np.pi, SAMPLES, endpoint=False)

rng = np.random.default_rng()

core_repeat = 2
core_t = np.repeat(TS, core_repeat)
core_size = rng.integers(2, 4, size=core_t.size)
core_bias = rng.uniform(-0.5, 0.5, size=core_t.size)

sat = rng.uniform(0.35, 0.9, size=core_t.size)
core_r = np.full(core_t.size, 255, dtype=int)
core_g = (60 + (1.0 - sat) * 160).astype(int)
core_b = (90 + (1.0 - sat) * 100).astype(int)
core_a = rng.integers(150, 231, size=core_t.size)
core_rgba = np.stack([core_r, core_g, core_b, core_a], axis=1)

# Glitter 
glitter_t = TS[::6]
glitter_size = rng.integers(1, 3, size=glitter_t.size)
glitter_phase = rng.uniform(0.0, 2.0 * np.pi, size=glitter_t.size)
glitter_orbit = rng.uniform(-1.4, 1.4, size=glitter_t.size)

sat_g = rng.uniform(0.5, 0.8, size=glitter_t.size)
glit_r = np.full(glitter_t.size, 255, dtype=int)
glit_g = (80 + (1.0 - sat_g) * 110).astype(int)
glit_b = (120 + (1.0 - sat_g) * 100).astype(int)
glitter_rgb = np.stack([glit_r, glit_g, glit_b], axis=1)
core_x, core_y = heart_xy(core_t)
glit_x, glit_y = heart_xy(glitter_t)

time_s = 0.0
running = True
while running:
    # events
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
            running = False

    #timing / heartbeat
    dt = clock.get_time() / 1000.0
    time_s += dt
    base_scale = 24.0
    beat = 1.0 + 0.08 * np.sin(time_s * 2.6) + 0.04 * np.sin(time_s * 5.2)
    scale = float(base_scale * beat)

    screen.fill((0, 0, 0, 255))
    layer = pygame.Surface((W, H), pygame.SRCALPHA)

    core_push = 1.0 + core_bias * 0.05
    core_px = (CENTER[0] + core_x * core_push * scale).astype(int)
    core_py = (CENTER[1] - core_y * core_push * scale).astype(int)
    for (px, py, sz, rgba) in zip(core_px, core_py, core_size, core_rgba):
        pygame.draw.circle(layer, rgba, (int(px), int(py)), int(sz))

    # Glitter dots 
    glit_px = (CENTER[0] + glit_x * (1.0 + glitter_orbit * 0.06) * scale).astype(int)
    glit_py = (CENTER[1] - glit_y * (1.0 + glitter_orbit * 0.06) * scale).astype(int)
    glit_a = (128 + 127 * np.cos(glitter_phase + time_s * 6.0)).astype(int).clip(0, 255)
    for (px, py, sz, rgb, a) in zip(glit_px, glit_py, glitter_size, glitter_rgb, glit_a):
        color = (int(rgb[0]), int(rgb[1]), int(rgb[2]), int(a))
        pygame.draw.circle(layer, color, (int(px), int(py)), int(sz))

    screen.blit(layer, (0, 0))
    pygame.display.flip()
    clock.tick(90)

pygame.quit()
