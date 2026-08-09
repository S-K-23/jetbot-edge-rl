# JetBot RL: obstacle + cliff avoidance

An RL pipeline that teaches a JetBot to avoid obstacles and table edges from
camera + distance-sensor input. The policy is trained entirely in a PyBullet
simulation (real-world RL is too slow and too destructive to train live),
then exported to ONNX and run on the JetBot's Jetson Nano for inference only.

## Architecture

```
sim/jetbot_sim_env.py  --training-->  train.py  --export-->  export_policy.py  --deploy-->  deploy/jetbot_inference.py
     (PyBullet)                       (PPO / SB3)              (-> ONNX)                     (runs on the JetBot)
```

Training and export run on a PC/laptop/cloud machine, since the Nano's GPU is
sized for inference, not for training a CNN-based policy. Only
`deploy/jetbot_inference.py` and its ONNX file ever touch the robot itself;
it has no dependency on PyBullet, PyTorch, or Stable-Baselines3.

## sim/jetbot_sim_env.py -- the simulation

A custom Gymnasium `Env` built directly on the PyBullet Python API (not a
higher-level robotics framework), because the whole scene here is simple
enough that a thin custom wrapper is easier to reason about than adopting a
larger simulation stack.

**Scene.** A flat square platform (`PLATFORM_HALF_SIZE`) with random box
obstacles scattered on it. Past the platform's edge there is no collision
geometry at all -- that absence is what creates the cliff hazard, standing in
for a table edge or step. The robot itself is a plain box body, not a
wheeled URDF: it's driven kinematically (`p.resetBaseVelocity`) from a
forward/angular speed computed per action, rather than through simulated
wheel motors and friction. This trades wheel-level physical fidelity for a
much faster and simpler-to-tune model -- see "Known limitations" below.

**Observation space** (`Dict`, matches SB3's `MultiInputPolicy` input format
with no extra glue code needed):
| key | shape | meaning |
|---|---|---|
| `image` | `(64, 64, 3)` uint8 | forward camera, rendered via PyBullet's `ER_TINY_RENDERER` (CPU-only, no GPU/display needed -- this is what makes headless training possible) |
| `obstacle_dist` | `(1,)` float32, `[0,1]` | forward raycast; `1.0` = nothing within `MAX_OBSTACLE_RANGE` |
| `cliff_dist` | `(1,)` float32, `[0,1]` | downward raycast from the front of the chassis; `1.0` = no floor found within `MAX_CLIFF_RANGE` |

**Action space.** `Discrete(5)`: forward, turn left, turn right, backward,
stop. Chosen over continuous motor control because it's simpler to learn,
maps directly onto the convenience methods JetBot's own `Robot` class already
exposes (`.forward()`, `.left()`, etc.), and matches how most JetBot builds
are already driven.

**Reward.** Forward progress each step, a small constant step penalty (to
discourage spinning in place), and a large negative terminal penalty on
collision or cliff detection. A z-position fallback check also terminates the
episode if the robot ever actually falls, in case the cliff raycast missed --
belt-and-suspenders, since the raycast is meant to catch the hazard *before*
it happens.

**Domain randomization**, applied on every `reset()`: obstacle/floor color,
obstacle size/count/position, robot start position and heading, and Gaussian
noise on both sensor readings. This is what the policy needs in order to
generalize past the exact pixel values and distances it saw in one scene.

**Multi-instance correctness.** Every PyBullet call in this file passes an
explicit `physicsClientId`. PyBullet's module-level functions default to
client 0 if you don't specify one, which silently corrupts training the
moment more than one `JetBotSimEnv` shares a process -- which is exactly
what SB3's default vectorized environment does. This was an actual bug
caught while building this, not a hypothetical one.

## train.py -- PPO training

Builds a vectorized stack of `JetBotSimEnv` instances and trains a PPO agent
(Stable-Baselines3) using `MultiInputPolicy`. That policy class handles the
`Dict` observation automatically: a small CNN (`NatureCNN`, via SB3's
`CombinedExtractor`) processes the image, the two sensor scalars pass through
flattened, the two streams are concatenated, and a small MLP (`mlp_extractor`)
turns the combined features into the actor/critic latents.

PPO (on-policy) was chosen over an off-policy alternative like SAC or DQN
specifically *because* training happens in simulation: sample efficiency
matters far less than it would on real hardware, and PPO's stability and
minimal hyperparameter sensitivity make it a reasonable default here.
Checkpoints are saved periodically via `CheckpointCallback`, and progress
logs to TensorBoard.

## export_policy.py -- ONNX export

Only the policy half of the actor-critic network is exported -- the value
function is training-only. `OnnxablePolicy` reproduces
`ActorCriticPolicy.forward()` up through the action logits
(`extract_features -> mlp_extractor -> action_net`), skipping the value head
and the sampling distribution, since deployment wants a deterministic
`argmax` over logits rather than a stochastic action draw. It calls the
policy's own `extract_features()` rather than reimplementing observation
preprocessing, so image normalization stays identical to what happened
during training.

The export explicitly uses PyTorch's legacy TorchScript-based exporter
(`dynamo=False`) rather than the newer default. That produces a simpler ONNX
graph that's far more likely to run on the old onnxruntime builds available
for the Nano's JetPack, and avoids an extra dependency (`onnxscript`) that
the newer exporter needs.

This file was tested against a real trained model: the ONNX output was
verified to match the PyTorch model's output exactly, both on a dummy input
and on a real observation pulled from the simulator.

## deploy/jetbot_inference.py -- on-robot inference

The only file meant to run on the Jetson Nano. Loads the ONNX policy via
`onnxruntime`, reads the camera and sensors, and drives the motors --
nothing here imports PyBullet, PyTorch, or Stable-Baselines3.

A few things worth knowing about how it's built:

- **Color/format conversion.** `camera.value` from the `jetbot` package comes
  back BGR (OpenCV convention); training used RGB frames from PyBullet, so
  `preprocess_frame()` converts color order, resizes to 64x64, and transposes
  to channel-first before handing the frame to the model.
- **Sensor reading is a placeholder you'll need to adapt.**
  `read_obstacle_cm()` assumes a Sharp-style analog IR sensor read through an
  ADS1115 ADC (the common setup on a Nano, which has no analog input pins of
  its own) -- swap this out if your sensor is wired differently.
- **You have one sensor; this design wants two.** A single forward-facing
  sensor covers obstacle detection well, but cliff detection generally needs
  a second, *downward-angled* sensor -- a forward sensor can't see a drop
  below the robot. Until that second sensor is wired up,
  `read_cliff_normalized()` returns a constant "floor detected, all clear"
  value, meaning the deployed robot will avoid obstacles but will not
  reliably detect edges.
- **Hard safety override, independent of the policy.**
  `apply_safety_override()` force-stops or backs up whenever the raw sensor
  reading crosses a critical threshold, regardless of what the network
  output. A policy trained in simulation is treated as "usually makes good
  choices," not as something to trust unconditionally around real hazards.

This file could not be tested against real hardware -- everything upstream
of it (environment, training, export) was built and verified to run for
real; this one is reasoned through carefully but unverified on a robot.
