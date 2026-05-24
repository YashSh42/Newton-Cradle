import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pymunk
import numpy as np
import pygame
import random
import math
import csv
import time
import cv2  


# =================== CONSTANTS ===================

WIDTH, HEIGHT      = 1000, 1000      # Physics simulation canvas
RENDER_W, RENDER_H  = 512, 512      # Output video resolution
FPS_STEP           = 1 / 60        # Simulation timestep (60 Hz)
TOTAL_STEPS        = 120           # 120 steps = 2 second simulation at 60 Hz
SAMPLE_RATE        = 1             # Render every single step
MIN_SPEED          = 120           # Minimum high-velocity speed threshold
MAX_SPEED          = 240           # Maximum high-velocity speed threshold
MAX_RETRIES        = 50            # Retries allowed if an anomaly occurs
EARLY_COLLISION_GUARD = 15         # Minimum steps before collision is allowed to occur
OUTPUT_DIR         = "dataset2"    # Output directory to store the generated dataset 

# Target restitution values with discrete 0.1 gaps
RESTITUTION_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TARGET_SAMPLES_PER_VAL = 300       # 11 bins * 300 = 3300 total samples

# ==================== CREATE DIRECTORIES ====================
def setup_output_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "samples"), exist_ok=True)

    # Initialize labels CSV
    labels_path = os.path.join(OUTPUT_DIR, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "restitution", "final_vel_ax", "final_vel_bx"])

    # Initialize scene metadata CSV
    meta_path = os.path.join(OUTPUT_DIR, "scene_metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "restitution",
            "shape_a", "shape_b",
            "mass_a", "mass_b",
            "speed_a", "speed_b",
            "initial_pos_ax", "initial_pos_ay",
            "initial_pos_bx", "initial_pos_by",
            "initial_vel_ax", "initial_vel_ay",
            "initial_vel_bx", "initial_vel_by",
            "final_vel_ax", "final_vel_bx",
            "collision_step"
        ])

# ==================== SIMULATION SETUP ==================== 
def create_space():
    space = pymunk.Space()
    space.gravity = (0, 0)  # No gravity
    space.damping = 1.0     # Vacuum environment (no air resistance / velocity damping)
    return space
# ==================== OBJECT CREATION ====================
def create_objects(space, target_e):
    params = {}
    
    # Positioning and velocity are set such that to ensure object A and B have head on collision
    
    pos_a = (random.randint(150, 250), 400) # Starting position of A
    pos_b = (random.randint(500, 600), 400) # Starting position of B

    # Random speeds for both objects
    speed_a = random.randint(MIN_SPEED, MAX_SPEED)
    speed_b = random.randint(MIN_SPEED, MAX_SPEED)

    mass_a, mass_b = 1.0, 1.0
    
    vel_ax, vel_ay = float(speed_a), 0.0
    vel_bx, vel_by = -float(speed_b), 0.0

    
    s_a = random.randint(100, 120)  # Size of Object A
    body_a = pymunk.Body(mass_a, float('inf'))  # Infinite moment of inertia to prevent rotation
    shape_a = pymunk.Poly.create_box(body_a, (s_a, s_a))
    
    body_a.position = pos_a
    body_a.velocity = (vel_ax, vel_ay)
    shape_a.elasticity = target_e
    shape_a.friction = 0.0
    shape_a.collision_type = 1
    space.add(body_a, shape_a) # Add object A to simulation space
    
    s_b = random.randint(100, 120) # Size of Object B
    body_b = pymunk.Body(mass_b, float('inf'))  # Infinite moment of inertia to prevent rotation
    shape_b = pymunk.Poly.create_box(body_b, (s_b, s_b))
    
    body_b.position = pos_b
    body_b.velocity = (vel_bx, vel_by)
    shape_b.elasticity = target_e
    shape_b.friction = 0.0
    shape_b.collision_type = 2
    space.add(body_b, shape_b)  # Add object B to simulation space
    # params stores all the details about the initial conditions of the simulation to record it a metadata CSV file
    params.update({
        "shape_a": "square", "shape_b": "square",
        "mass_a": mass_a, "mass_b": mass_b,
        "speed_a": speed_a, "speed_b": speed_b,
        "initial_pos_ax": pos_a[0], "initial_pos_ay": pos_a[1],
        "initial_pos_bx": pos_b[0], "initial_pos_by": pos_b[1],
        "initial_vel_ax": vel_ax, "initial_vel_ay": vel_ay,
        "initial_vel_bx": vel_bx, "initial_vel_by": vel_by
    })
    
    return body_a, shape_a, body_b, shape_b, params

# Checks for collision between object A and B and returns when the collision happens in the simulation
def register_collision_handler(space, shape_a, shape_b):
    collision_info = {"happened": False}
    def on_collision(arbiter, space, data):
        collision_info["happened"] = True
        
    space.on_collision(
        shape_a.collision_type, 
        shape_b.collision_type, 
        begin=on_collision
    )
    return collision_info

# ==================== FRAME RENDERING ====================
def render_frame(surface, body_a, shape_a, body_b, shape_b):
    
    surface.fill((255, 255, 255))

    def scale(x, y):
        x_render = int(x * RENDER_W / WIDTH)
        y_render = int(y * RENDER_H / HEIGHT)
        return x_render, y_render

    def draw_object(body, shape):
        local_vertices = shape.get_vertices()
        angle = body.angle
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        world_corners = []
        for vx, vy in local_vertices:
            # 2D Rigid Rotation Transformation
            world_x = body.position.x + (cos_a * vx - sin_a * vy)
            world_y = body.position.y + (sin_a * vx + cos_a * vy)
            
            rx, ry = scale(world_x, world_y)
            world_corners.append((rx, ry))
            
        pygame.draw.polygon(surface, (0, 0, 0), world_corners)  # Draw squares as solid black

    draw_object(body_a, shape_a)
    draw_object(body_b, shape_b)

    raw = pygame.image.tobytes(surface, 'RGB') # Get raw RGB pixel data from the surface
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(RENDER_H, RENDER_W, 3)  # frame has been rendered in RGB format
    return frame

# ==================== SIMULATION EXECUTION ====================
def run_simulation(space, body_a, shape_a, body_b, shape_b, collision_info):
    surface = pygame.Surface((RENDER_W, RENDER_H))
    all_frames = []
    collision_step = None
    discard_reason = None
    final_vel_ax, final_vel_bx = 0.0, 0.0

    for step in range(TOTAL_STEPS):
        space.step(FPS_STEP)

        pos_a = body_a.position
        pos_b = body_b.position

        # Containment check to discard objects exiting canvas bounds too quickly
        if (pos_a.x < 0 or pos_a.x > WIDTH or pos_a.y < 0 or pos_a.y > HEIGHT or
            pos_b.x < 0 or pos_b.x > WIDTH or pos_b.y < 0 or pos_b.y > HEIGHT):
            discard_reason = "out_of_bounds"
            break

        if collision_info["happened"] and collision_step is None:
            collision_step = step

        frame = render_frame(surface, body_a, shape_a, body_b, shape_b)
        all_frames.append(frame)

    if discard_reason is None:
        final_vel_ax = float(body_a.velocity.x)
        final_vel_bx = float(body_b.velocity.x)

    return all_frames, collision_step, discard_reason, final_vel_ax, final_vel_bx

# ==================== DATA VERIFICATION ====================
def log_discard(sample_id, reason, details=""):
    print(f"[DISCARD] sample_{sample_id:05d} | reason: {reason:<20} | {details}")

def check_discard(sample_id, collision_happened, collision_step, discard_reason):
    if discard_reason == "out_of_bounds":
        log_discard(sample_id, "out_of_bounds", "Object exited the arena.")
        return True

    if collision_step is not None and collision_step < EARLY_COLLISION_GUARD:
        log_discard(sample_id, "early_collision", f"Impact at step {collision_step} is too early.")
        return True

    if not collision_happened:
        log_discard(sample_id, "no_collision", "Head-on trajectory failed to strike.")
        return True

    return False

# ==================== VIDEO COMPILATION & SAVE ====================
def save_sample(sample_id, frames, target_e, label_writer, meta_writer, collision_step, params, final_vel_ax, final_vel_bx):
    sample_name = f"sample_{sample_id:05d}"
    
    # Export full rendering timeline directly as an MP4 Video
    video_path = os.path.join(OUTPUT_DIR, "samples", f"{sample_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (RENDER_W, RENDER_H))
    
    for frame in frames:
        # Convert RGB back to BGR for correct OpenCV color mapping
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)
    video_writer.release()

    # Append targets & output telemetry to master labels index
    label_writer.writerow([sample_name, float(target_e), final_vel_ax, final_vel_bx])

    # Record complete initial/final parameters to CSV metadata
    meta_writer.writerow([
        sample_name,
        float(target_e),
        params.get("shape_a", "square"),
        params.get("shape_b", "square"),
        params.get("mass_a", 1.0),
        params.get("mass_b", 1.0),
        params.get("speed_a", 0.0),
        params.get("speed_b", 0.0),
        params.get("initial_pos_ax", 0.0),
        params.get("initial_pos_ay", 400.0),
        params.get("initial_pos_bx", 0.0),
        params.get("initial_pos_by", 400.0),
        params.get("initial_vel_ax", 0.0),
        params.get("initial_vel_ay", 0.0),
        params.get("initial_vel_bx", 0.0),
        params.get("initial_vel_by", 0.0),
        final_vel_ax,
        final_vel_bx,
        collision_step if collision_step is not None else ""
    ])

# =================== PROGRESS TRACKING ===================
def print_progress(counts, start_time):
    total_done = sum(counts.values())
    total_target = len(RESTITUTION_VALUES) * TARGET_SAMPLES_PER_VAL
    elapsed = time.perf_counter() - start_time

    progress_strs = [f"e={val:.1f}: {counts[val]}/{TARGET_SAMPLES_PER_VAL}" for val in RESTITUTION_VALUES]
    line = " | ".join(progress_strs)
    print(f"\r{line} | total: {total_done}/{total_target} | {elapsed:.1f}s", end="")

# =================== MAIN EXECUTION ===================
def main():
    start_time = time.perf_counter()
    setup_output_dirs() # Create the output directories
    pygame.init()   # Initialize Pygame

    labels_path = os.path.join(OUTPUT_DIR, "labels.csv")
    meta_path   = os.path.join(OUTPUT_DIR, "scene_metadata.csv")

    counts = {val: 0 for val in RESTITUTION_VALUES}
    sample_id = 0

    with open(labels_path, "a", newline="") as lf, \
         open(meta_path,   "a", newline="") as mf:

        label_writer = csv.writer(lf)
        meta_writer  = csv.writer(mf)

        while any(counts[val] < TARGET_SAMPLES_PER_VAL for val in RESTITUTION_VALUES):
            # Select a restitution value that still needs samples
            incomplete_vals = [val for val in RESTITUTION_VALUES if counts[val] < TARGET_SAMPLES_PER_VAL]
            target_e = random.choice(incomplete_vals)

            success = False
            for attempt in range(MAX_RETRIES):
                space = create_space()
                body_a, shape_a, body_b, shape_b, params = create_objects(space, target_e)
                collision_info = register_collision_handler(space, shape_a, shape_b)

                all_frames, collision_step, discard_reason, final_vel_ax, final_vel_bx = run_simulation(
                    space, body_a, shape_a, body_b, shape_b, collision_info
                )

                collision_happened = collision_info["happened"]

                if check_discard(sample_id, collision_happened, collision_step, discard_reason):
                    continue

                # Ensure we have rendered frames
                if len(all_frames) == 0:
                    log_discard(sample_id, "empty_simulation", "No simulation steps recorded.")
                    continue

                save_sample(
                    sample_id, all_frames, target_e,
                    label_writer, meta_writer, collision_step, params,
                    final_vel_ax, final_vel_bx
                )

                counts[target_e] += 1
                sample_id += 1
                print_progress(counts, start_time)

                success = True
                break

            if not success:
                log_discard(
                    sample_id, "max_retries",
                    f"Restitution target e={target_e:.1f} exhausted all {MAX_RETRIES} attempts."
                )

    print(f"\n\nDataset generation complete.")
    print(f"Total samples saved : {sample_id}")
    print(f"Time taken          : {time.perf_counter() - start_time:.2f}s")
    print(f"Output directory    : {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()