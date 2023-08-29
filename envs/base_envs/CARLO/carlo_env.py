import numpy as np
from .world import World
from .agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian
from .geometry import Point, Line
import time
# from tkinter import *
import gym
import gym.spaces as spaces
from moviepy.video.io.bindings import mplfig_to_npimage


class CarloEnv:
    def __init__(self, continuous_actions=True):
        dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
        world_width = 120/2 # in meters
        world_height = 120/2
        self.inner_building_radius = 30/2
        num_lanes = 2
        self.lane_marker_width = 0.5
        num_of_lane_markers = 52/2
        self.lane_width = 6# 3.5
        self.continuous_actions = continuous_actions
        self.ppm = 20

        w = World(dt, width = world_width, height = world_height, ppm = self.ppm) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

        # Let's add some sidewalks and RectangleBuildings.
        # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks / zebra crossings / or creating lanes.
        # A CircleBuilding or RingBuilding object is also static -- they do not move. But as opposed to Painting, they can be collided with.

        # To create a circular road, we will add a CircleBuilding and then a RingBuilding around it
        cb = CircleBuilding(Point(world_width/2, world_height/2), self.inner_building_radius, 'gray80')
        w.add(cb)
        rb = RingBuilding(Point(world_width/2, world_height/2), self.inner_building_radius + num_lanes * self.lane_width + (num_lanes - 1) * self.lane_marker_width, 1+np.sqrt((world_width/2)**2 + (world_height/2)**2), 'gray80')
        w.add(rb)

        # w.add(CircleBuilding(Point(72.5, 107.5), Point(95, 25))) 


        # Let's also add some lane markers on the ground. This is just decorative. Because, why not.
        self.waypoints = []
        for lane_no in range(num_lanes - 1):
            lane_markers_radius = self.inner_building_radius + (lane_no + 1) * self.lane_width + (lane_no + 0.5) * self.lane_marker_width
            lane_marker_height = np.sqrt(2*(lane_markers_radius**2)*(1-np.cos((2*np.pi)/(2*num_of_lane_markers)))) # approximate the circle with a polygon and then use cosine theorem
            for i, theta in enumerate(np.arange(0, 2*np.pi, 2*np.pi / num_of_lane_markers)):
                dx = lane_markers_radius * np.cos(theta)
                dy = lane_markers_radius * np.sin(theta)
                if (i % 13 == 0) & (i > -1):
                    self.waypoints.append(
                        CircleBuilding(Point(world_width/2 + dx + (0 * self.lane_width) * np.cos(theta) , world_height/2 + dy + (0 * self.lane_width) * np.sin(theta)), 5, 'blue')
                        # Painting(Point(world_width/2 + dx + (-.5 * self.lane_width) * np.cos(theta) , world_height/2 + dy + (-.5 * self.lane_width) * np.sin(theta)), Point(self.lane_marker_width, lane_marker_height), 'red', heading = theta)
                        )
                    wp = self.waypoints[-1]
                    wp.collidable = False
                    w.add(wp)
                # w.add(Painting(Point(world_width/2 + dx, world_height/2 + dy), Point(self.lane_marker_width, lane_marker_height), 'white', heading = theta))
        
        # Let's also add some lane markers on the ground. This is just decorative. Because, why not.
        # self.waypoints = self.waypoints[[]]

        # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
        self.world_width = world_width
        self.world_height = world_height
        self.center = np.array([30, 30])
        self.dt = dt
        self.world = w
        self.reset()
        self.world.render() # This visualizes the world we just constructed.

        # gym environment specific variables
        if continuous_actions:
            self.action_space = spaces.Box(-1., 1., shape=(2,), dtype=np.float32)
        else:
            # up, right, down, left, nothing
            self.action_space = spaces.Discrete(30)

        
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.get_state().shape, dtype='float32')

    def reset(self):
        try:
            self.world.visualizer.close()
        except:
            pass

        self.world.reset()
        
        self.current_wp = 0

        xs = np.linspace(np.pi/2, 2*np.pi+np.pi/2, 10)
        theta = np.random.choice(xs) % (np.pi*2)
        lane_markers_radius = self.inner_building_radius + (0 + 1) * self.lane_width + (0 + 0.5) * self.lane_marker_width
        dx = lane_markers_radius * np.cos(theta)
        dy = lane_markers_radius * np.sin(theta)
        x = self.world_width/2 + dx
        y = self.world_height/2 + dy

        c1 = Car(Point(x, y), theta + np.pi/2)
        c1.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
        c1.min_speed = 3.0 # let's say the maximum is 30 m/s (108 km/h)
        c1.velocity = Point(0.0, 0.0)
        self.agent = c1
        self.world.add(c1)

        self.state = self.get_state()
        return self.state, {}
    
    def unnormalize(self, states):
        return states * np.array([self.world_width, self.world_height, self.agent.max_speed, self.agent.max_speed, 2*np.pi]) 

    def get_state(self):
        return np.array([self.agent.x, self.agent.y, self.agent.xp, self.agent.yp, self.agent.heading]) / np.array([self.world_width, self.world_height, self.agent.max_speed, self.agent.max_speed, 2*np.pi]) 

    def label(self, state):
        signal, labels = {}, {}
        
        # import pdb; pdb.set_trace()
        for idx, distance in enumerate(self.distance_to_waypoints(state)):
            if distance <= 5:
                key = 'wp_%s' % idx
                labels.update({key: 1})
        # if len(labels): import pdb; pdb.set_trace()
        # if any(self.distance_to_waypoints(state) < 2):
        #     labels.update({'wp': 1})
        # if self.distance_to_waypoints(state)[self.current_wp] < 2:
        #     labels.update({'wp': 1})
        
        if self.world.collision_exists():
            labels.update({'crash':1})
        # for circle, color in self.circles:
        #     val = np.linalg.norm(state - circle[:-1])
        #     if val < circle[-1]:
        #         labels.update({color:1})
        #         val = -1
            
        #     signal.update({color: -val})
        return labels, signal
    
    def distance_to_waypoints(self, state):
        new_state = state * np.array([self.world_width, self.world_height, self.agent.max_speed, self.agent.max_speed, 2*np.pi]) 
        return np.array([np.linalg.norm([new_state[0] - wp.x, new_state[1] - wp.y]) for wp in self.waypoints])

    def render(self, states = [], save_dir=None, save_states=False):
        
        if not self.world.headless:
            self.world.render()

            ppm = self.world.visualizer.ppm
            dh = self.world.visualizer.display_height
            if len(states):
                for (x1,y1), (x2, y2) in zip(np.array(states)[:-1, 0:2], np.array(states)[1:, 0:2]):
                    #self.world.add(Line(Point(x1,y1), Point(x2, y2)))
                    x1 *= self.world_width
                    x2 *= self.world_width
                    y1 *= self.world_height
                    y2 *= self.world_height
                    self.world.visualizer.win.plot_line_(ppm*x1, dh - ppm*y1, ppm*x2, dh - ppm*y2, fill='red', width='2')
                # coords = np.array(states)[0:2]
                # coords[:, 0] *= ppm
                # coords[:, 1] = dh - ppm*coords[:, 1]
                # self.world.visualizer.win.plot_line(coords.T.flatten().tolist(), color='green')
            #import pdb; pdb.set_trace()
            self.world.render()

            if save_dir is not None:
                self.world.visualizer.save_fig(save_dir)
                self.world.remove_agents()
                #self.world.visualizer.save_fig(save_dir)
    
            if save_states:
                np.save(save_dir + '.npy', np.array(states))
         
        # numpy_fig = mplfig_to_npimage(self.fig)  # convert it to a numpy array
        # return numpy_fig
    
    def step(self, action):

        if not self.continuous_actions:
            if action == 0:
                u = np.array([[0, .5]])
            elif action == 1:
                u = np.array([[.5, 0]])
            elif action == 2:
                u = np.array([[0, -.5]])
            elif action == 3:
                u = np.array([[-.5, 0]])
            elif action == 4:
                u = np.array([[0, 0]])
            else:
                raise NotImplemented
        else:
            u = action

        self.agent.set_control(u[0], u[1])
        # self.agent.set_control(1, .2)
        self.world.tick() # This ticks the world for one time step (dt second)
        # if self.world.collision_exists():
        #     reward = -500
        # else:
        reward = (-1 * np.linalg.norm(u)) #* 0.25
        position = np.array([self.agent.x, self.agent.y])
        stay_centered_reward = (21.25 - np.linalg.norm(position - self.center)) / self.inner_building_radius  # negative penalty for distance from the center of the track
        terminated = self.world.collision_exists()
        self.state = self.get_state()
        # if (np.linalg.norm(self.state[2:4]*self.agent.max_speed)+1e-7) < self.agent.min_speed:
        #     import pdb; pdb.set_trace()
        # if self.distance_to_waypoints(self.state) < 2:

        # Reward is higher if the agent is closer to the center of the map.
        # center: self.world_width/2, self.world_height/2
        center_x, center_y = self.world_width/2, self.world_height/2
        circle_center_reward = np.square(self.x - center_x) + np.square(self.y - center_y)

        # return self.state, reward, terminated, {"rho": np.array(0)}
        return self.state, circle_center_reward, terminated, {"rho": np.array(0)}
        
