
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
 
 
IMG_SIZE = 64
MAX_OBSTACLE_RANGE = 0.60   
MAX_CLIFF_RANGE = 0.15
PLATFORM_HALF_SIZE = 1.2
MAX_EPISODE_STEPS = 500
CONTROL_HZ = 10
PHYSICS_HZ = 60
 
 
class JetBotSimEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
 
    def __init__(self, render_mode=None, domain_randomization=True, n_obstacles=6):
        super().__init__()
        self.render_mode = render_mode
        self.domain_randomization = domain_randomization
        self.n_obstacles = n_obstacles
 
        self.client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
 
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8),
            "obstacle_dist": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "cliff_dist": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })
 
        self._robot = None
        self._obstacle_ids = []
        self._platform_id = None
        self._step_count = 0
        self._prev_xy = np.zeros(2)
 
    # Gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(1.0 / PHYSICS_HZ, physicsClientId=self.client)
 
        self._build_scene()
        self._step_count = 0
        pos, _ = p.getBasePositionAndOrientation(self._robot, physicsClientId=self.client)
        self._prev_xy = np.array(pos[:2])
 
        obs = self._get_obs()
        return obs, {}
 
    def step(self, action):
        self._apply_action(int(action))
 
        for _ in range(PHYSICS_HZ // CONTROL_HZ):
            p.stepSimulation(physicsClientId=self.client)
 
        obs = self._get_obs()
        pos, _ = p.getBasePositionAndOrientation(self._robot, physicsClientId=self.client)
        xy = np.array(pos[:2])
 
        collided = self._check_collision()
        cliff = obs["cliff_dist"][0] >= 0.97
        fell = pos[2] < -0.3
 
        progress = float(np.linalg.norm(xy - self._prev_xy))
        self._prev_xy = xy
 
        reward = progress * 10.0
        reward -= 0.01
        terminated = False
        if collided:
            reward -= 5.0
            terminated = True
        if cliff or fell:
            reward -= 5.0
            terminated = True
 
        self._step_count += 1
        truncated = self._step_count >= MAX_EPISODE_STEPS
 
        info = {"collided": collided, "cliff": bool(cliff), "fell": bool(fell)}
        return obs, reward, terminated, truncated, info
 
    def close(self):
        if p.isConnected(physicsClientId=self.client):
            p.disconnect(physicsClientId=self.client)
 
    # Scene construction
    def _build_scene(self):
        half = PLATFORM_HALF_SIZE
 
        floor_color = (list(np.random.uniform(0.3, 0.9, 3)) + [1]) if self.domain_randomization else [0.6, 0.6, 0.65, 1]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half, half, 0.02], physicsClientId=self.client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, 0.02], rgbaColor=floor_color, physicsClientId=self.client)
        self._platform_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[0, 0, -0.02], physicsClientId=self.client,
        )
 
        # Robot: a simple box body
        r_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.045, 0.03], physicsClientId=self.client)
        r_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.045, 0.03], rgbaColor=[0.1, 0.1, 0.1, 1], physicsClientId=self.client)
        start_xy = np.random.uniform(-half * 0.3, half * 0.3, 2) if self.domain_randomization else np.zeros(2)
        start_yaw = np.random.uniform(-np.pi, np.pi) if self.domain_randomization else 0.0
        quat = p.getQuaternionFromEuler([0, 0, start_yaw])
        self._robot = p.createMultiBody(
            baseMass=0.5, baseCollisionShapeIndex=r_col, baseVisualShapeIndex=r_vis,
            basePosition=[start_xy[0], start_xy[1], 0.03], baseOrientation=quat,
            physicsClientId=self.client,
        )
 
        # Obstacles
        self._obstacle_ids = []
        n = self.n_obstacles if self.domain_randomization else max(2, self.n_obstacles // 2)
        for _ in range(n):
            size = np.random.uniform(0.03, 0.08) if self.domain_randomization else 0.05
            pos_xy = np.random.uniform(-half * 0.8, half * 0.8, 2)
            if np.linalg.norm(pos_xy - start_xy) < 0.25:
                continue
            obs_color = (list(np.random.uniform(0.2, 0.9, 3)) + [1]) if self.domain_randomization else [0.8, 0.2, 0.2, 1]
            o_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size], physicsClientId=self.client)
            o_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=obs_color, physicsClientId=self.client)
            oid = p.createMultiBody(
                baseMass=0, baseCollisionShapeIndex=o_col, baseVisualShapeIndex=o_vis,
                basePosition=[pos_xy[0], pos_xy[1], size], physicsClientId=self.client,
            )
            self._obstacle_ids.append(oid)
 
    # Action -> motion
    def _apply_action(self, action):
        pos, orn = p.getBasePositionAndOrientation(self._robot, physicsClientId=self.client)
        yaw = p.getEulerFromQuaternion(orn)[2]
 
        forward_speed = 0.25
        turn_speed = 1.5    
 
        lin, ang = 0.0, 0.0
        if action == 0:
            lin = forward_speed
        elif action == 1:
            ang = turn_speed
        elif action == 2:
            ang = -turn_speed
        elif action == 3: 
            lin = -forward_speed
        elif action == 4: 
            pass
 
        vx = lin * np.cos(yaw)
        vy = lin * np.sin(yaw)
        p.resetBaseVelocity(
            self._robot, linearVelocity=[vx, vy, 0], angularVelocity=[0, 0, ang],
            physicsClientId=self.client,
        )
 
    # Sensing
    def _get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self._robot, physicsClientId=self.client)
        yaw = p.getEulerFromQuaternion(orn)[2]
 
        image = self._render_camera(pos, yaw)
        obstacle_dist = self._read_obstacle_sensor(pos, yaw)
        cliff_dist = self._read_cliff_sensor(pos, yaw)
 
        return {
            "image": image,
            "obstacle_dist": np.array([obstacle_dist], dtype=np.float32),
            "cliff_dist": np.array([cliff_dist], dtype=np.float32),
        }
 
    def _render_camera(self, pos, yaw):
        cam_offset = 0.06
        cam_pos = [pos[0] + cam_offset * np.cos(yaw), pos[1] + cam_offset * np.sin(yaw), pos[2] + 0.02]
        target = [cam_pos[0] + np.cos(yaw), cam_pos[1] + np.sin(yaw), cam_pos[2] - 0.05]
        view = p.computeViewMatrix(cam_pos, target, [0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=120, aspect=1.0, nearVal=0.02, farVal=3.0)
      
        _, _, rgba, _, _ = p.getCameraImage(
            IMG_SIZE, IMG_SIZE, view, proj,
            renderer=p.ER_TINY_RENDERER, flags=p.ER_NO_SEGMENTATION_MASK,
            physicsClientId=self.client,
        )
        rgb = np.reshape(rgba, (IMG_SIZE, IMG_SIZE, 4))[:, :, :3].astype(np.uint8)
        return rgb
 
    def _read_obstacle_sensor(self, pos, yaw):
        start = [pos[0], pos[1], pos[2]]
        end = [pos[0] + MAX_OBSTACLE_RANGE * np.cos(yaw), pos[1] + MAX_OBSTACLE_RANGE * np.sin(yaw), pos[2]]
        hit = p.rayTest(start, end, physicsClientId=self.client)[0]
        frac = hit[2]
        if self.domain_randomization:
            frac = float(np.clip(frac + np.random.normal(0, 0.02), 0.0, 1.0))
        return float(frac)
 
    def _read_cliff_sensor(self, pos, yaw):
        front = [pos[0] + 0.07 * np.cos(yaw), pos[1] + 0.07 * np.sin(yaw), pos[2]]
        below = [front[0], front[1], front[2] - MAX_CLIFF_RANGE]
        hit = p.rayTest(front, below, physicsClientId=self.client)[0]
        frac = hit[2] 
        if self.domain_randomization:
            frac = float(np.clip(frac + np.random.normal(0, 0.01), 0.0, 1.0))
        return float(frac)
 
    def _check_collision(self):
        for oid in self._obstacle_ids:
            if len(p.getContactPoints(self._robot, oid, physicsClientId=self.client)) > 0:
                return True
        return False
