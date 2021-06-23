### Chris Osgood, 5/27/21
# This is the 3D thrust vectored hover vehicle environment implementation
# Using OpenAI Gym to create a custom environment
import gym
from gym import spaces
import math
import numpy as np

class Hoverer3DEnv(gym.Env):
    def __init__(self):
        self.gravity = 9.81
        self.mass = 300
        self.moment_arm = 0.25
        self.I = np.diag([6, 6, 0.05])
        self.dt = 0.01
        self.ang_th = math.pi / 12 # Angle Threshold: 15 degrees each direction
        self.pos_th = 1 # Position Threshold: must stay within a 1x1x1 meter box
        self.vec_th = math.pi / 36 # Thrust Vectoring Threshold: 5 degrees each direction

        ## Observation (State) space is:
        #   x,y,z-pos   x,y,z-vel   x,y-angle   x,y-angleVel
        # Note: z-angle is always zero -- assuming no roll
        upperbound = np.array([2*self.pos_th,2*self.pos_th,2*self.pos_th, np.inf,np.inf,np.inf, 2*self.ang_th,2*self.ang_th, np.inf,np.inf])
        self.observation_space = spaces.Box(-upperbound, upperbound)

        ## Action space is:
        #   x-angleThrust    y-angleThrust    Throttle(%)
        #   [-5deg, 5deg]    [-5deg, 5deg]    [.75, 1.25]
        low = np.array([-self.vec_th, -self.vec_th, 0.75])
        high = np.array([self.vec_th, self.vec_th, 1.25])
        self.action_space = spaces.Box(low, high)

        # set initial state
        self.reset()

    def step(self, action):
        pos = np.zeros((3,))
        vel = np.zeros((3,))
        ang = np.zeros((3,))
        angvel = np.zeros((3,))
        ang[2] = 0 # z-value (roll) doesn't matter, except for the update for angvel
        angvel[2] = 0 # z-value (roll) doesn't matter, except for the update for angvel
        pos[0],pos[1],pos[2], vel[0],vel[1],vel[2], ang[0],ang[1], angvel[0],angvel[1] = self.state
        thrust_ang = np.zeros((3,))
        thrust_ang[2] = 0 # z-value (roll) doesn't matter, except for the update for angvel
        thrust_ang[0], thrust_ang[1], throttle = action
        phi = ang - thrust_ang
        #phi[1] += math.pi / 2
        thrust = self.mass * self.gravity * throttle
        #xyz forces = thrust * np.array([math.cos(phi[0])*math.cos(phi[1]), math.sin(phi[0])*math.cos(phi[1]), math.sin(phi[1])])
        #yxz2 forces = thrust * np.array([-math.cos(phi[1]), math.cos(phi[1])*math.sin(phi[0]), math.cos(phi[0])*math.cos(phi[1])])
        forces = thrust * np.array([-math.sin(phi[1]), math.cos(phi[1])*math.sin(phi[0]), math.cos(phi[0])*math.cos(phi[1])])
        forces[2] -= self.gravity * self.mass
        #print("Forces: ", forces)
        # make sure torque is r cross F (could do this using np.cross()
        torques = np.zeros((3,))
        torques[0] = forces[1] * self.moment_arm
        torques[1] = -forces[0] * self.moment_arm
        torques[2] = 0 # no torque about z-axis
        #print("Torques: ", torques)

        # Update/Step forward in time
        pos += self.dt * vel
        vel += self.dt * forces / self.mass
        ang += self.dt * angvel
        angvel += self.dt * (np.linalg.inv(self.I) @ (torques - np.cross(angvel, self.I @ angvel)))
        # update state
        self.state = np.array([pos[0],pos[1],pos[2], vel[0],vel[1],vel[2], ang[0],ang[1], angvel[0],angvel[1]])

        self.time_left -= self.dt # reduce time left by the time we stepped

        done = bool( # set done boolean
            self.time_left <= 0 or
            np.any(pos < -self.pos_th) or np.any(pos > self.pos_th) or
            np.any(ang < -self.ang_th) or np.any(ang > self.ang_th)
        )

        # set rewards
        if done:
            reward = -100
        if not done:
            reward = 1

        info = {} # set info

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        """
        Initialize position randomly within a 0.1m box around the center.
        Initialize velocity and angular velocity to zero.
        Initialize angle randomly within 1 degree box.
        """
        pos = -0.05 + 0.1 * np.random.uniform(size=(3,)) # shape (3,)
        ang = -(math.pi / 360) + (math.pi / 180) * np.random.uniform(size=(2,)) # shape (2,)
        self.state = np.array([pos[0],pos[1],pos[2], 0,0,0, ang[0],ang[1], 0,0])
        self.time_left = 4
        return self.state
