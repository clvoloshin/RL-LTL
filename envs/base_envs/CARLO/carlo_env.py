import numpy as np
from .world import World
from .agents import Car, RingBuilding, CircleBuilding, Painting, Pedestrian, RectangleBuilding
from .geometry import Point, Line
import time
# from tkinter import *
import gym
import gym.spaces as spaces
from moviepy.video.io.bindings import mplfig_to_npimage


class CarloEnv:
    def __init__(self, continuous_actions=True):
        dt = 0.1 # time steps in terms of seconds. In other words, 1/dt is the FPS.
        world_width = 160/2 # in meters
        world_height = 100/2
        self.inner_building_radius = 11/2
        # num_lanes = 2
        self.lane_marker_width = 0.75
        # num_of_lane_markers = 52/2
        self.lane_width = 5.5# 3.5
        self.continuous_actions = continuous_actions
        self.ppm = 20
        self.border_radius = 2.5
        self.waypoint_radius = 3.0
        self.render_live = False

        w = World(dt, width = world_width, height = world_height, ppm = self.ppm) # The world is 120 meters by 120 meters. ppm is the pixels per meter.

        # Let's add some sidewalks and RectangleBuildings.
        # A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks / zebra crossings / or creating lanes.
        # A CircleBuilding or RingBuilding object is also static -- they do not move. But as opposed to Painting, they can be collided with.
        
        # To create a figure-eight road, we will add two CircleBuilding and then two RingBuildings around them
        cb1 = CircleBuilding(Point(world_width* (1/3), world_height/2), self.inner_building_radius, 'gray80')
        cb2 = CircleBuilding(Point(world_width* (2/3), world_height/2), self.inner_building_radius, 'gray80')
        self.cb1 = cb1
        self.cb2 = cb2
        w.add(cb1)
        w.add(cb2)
        toprect = RectangleBuilding(Point(world_width /2, world_height - self.border_radius), Point(world_width, self.border_radius * 2), 'gray80')
        bottomrect = RectangleBuilding(Point(world_width / 2., self.border_radius), Point(world_width, self.border_radius * 2), 'gray80')
        leftrect = RectangleBuilding(Point(self.border_radius, world_height / 2), Point(self.border_radius * 2, world_height - (self.border_radius * 4)), 'gray80')
        rightrect = RectangleBuilding(Point(world_width - self.border_radius, world_height / 2), Point(self.border_radius * 2, world_height - (self.border_radius * 4)), 'gray80')
        w.add(toprect)
        w.add(bottomrect)
        w.add(leftrect)
        w.add(rightrect)
                
        self.distance_between_circles = abs(cb2.center.x - cb1.center.x) - 2 * self.inner_building_radius

        # let's add some potholes to make the road a little more interesting.
        self.potholes = []
        # self.pothole_height_offset = 12.
        # self.pothole_width_offset = 0
        # self.pothole_height = 3.5
        # self.pothole_width = 3.5
        # self.potholes.append(Painting(Point(world_width/2 + self.pothole_width_offset, world_height/2 + self.pothole_height_offset), Point(self.pothole_width, self.pothole_height), 'orange'))
        # self.potholes.append(Painting(Point(world_width/2 - self.pothole_width_offset, world_height/2 - self.pothole_height_offset), Point(self.pothole_width, self.pothole_height), 'orange'))
        # for pothole in self.potholes:
        #     pothole.collidable = False
        #     w.add(pothole) 
        # Let's also add some lane markers on the ground as waypoints for the LTL specification
        self.waypoints = []
        # for lane_no in range(num_lanes - 1):
        #     lane_markers_radius = self.inner_building_radius + (lane_no + 1) * self.lane_width + (lane_no + 0.5) * self.lane_marker_width
        #     lane_marker_height = 0.5 #np.sqrt(2*(lane_markers_radius**2)*(1-np.cos((2*np.pi)/(2*num_of_lane_markers)))) # approximate the circle with a polygon and then use cosine theorem
        #     lane_marker_width = self.lane_width * num_lanes
        #     for i, theta in enumerate(np.arange(0, 2*np.pi, 2*np.pi / num_of_lane_markers)):
        #         dx = lane_markers_radius * np.cos(theta)
        #         dy = lane_markers_radius * np.sin(theta)
        #         if (i % 13 == 0) & (i > -1):
        #             self.waypoints.append(
        #                 #CircleBuilding(Point(world_width/2 + dx + (0 * self.lane_width) * np.cos(theta) , world_height/2 + dy + (0 * self.lane_width) * np.sin(theta)), 5, 'blue')
        #                 Painting(Point(world_width/4 + dx + (0 * self.lane_width) * np.cos(theta) , world_height/2 + dy + (0 * self.lane_width) * np.sin(theta)), Point(lane_marker_width, lane_marker_height), 'blue', heading=theta)
        #                 )
        #             wp = self.waypoints[-1]
        #             wp.collidable = False
        #             w.add(wp)
                # w.add(Painting(Point(world_width/2 + dx, world_height/2 + dy), Point(self.lane_marker_width, lane_marker_height), 'white', heading = theta))
        lane_marker_height = 0.5 #np.sqrt(2*(lane_markers_radius**2)*(1-np.cos((2*np.pi)/(2*num_of_lane_markers)))) # approximate the circle with a polygon and then use cosine theorem
        # import pdb; pdb.set_trace()
        lane_marker_width = cb1.center.x - cb1.radius - self.border_radius * 2
        middle_width = cb2.center.x - cb1.center.x - 2 * self.inner_building_radius
        self.lane_marker_width = lane_marker_width
        # Let's also add some lane markers on the ground. This is just decorative. Because, why not.
        self.waypoints.append(Painting(Point(self.border_radius  * 2 + (lane_marker_width / 2.), world_height / 2.), Point(lane_marker_width, lane_marker_height), 'blue'))
        self.waypoints.append(Painting(Point(world_width / 2., world_height / 2.), Point(middle_width, lane_marker_height), 'blue'))
        self.waypoints.append(Painting(Point(world_width - (self.border_radius  * 2 + (lane_marker_width / 2.)), world_height / 2.),
                                       Point(lane_marker_width, lane_marker_height), 'blue'))
        # self.waypoints.append(Painting())
        # self.waypoints.append(CircleBuilding(Point(world_width* (1/6), world_height/2), self.waypoint_radius, 'blue'))
        # self.waypoints.append(CircleBuilding(Point(world_width* (5/6), world_height/2), self.waypoint_radius, 'blue'))
        for wp in self.waypoints:
            wp.collidable = False
            w.add(wp) 
        # A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
        self.world_width = world_width
        self.world_height = world_height
        self.center = np.array([self.world_width / 2, self.world_height / 2])
        self.relative_center = np.array([0.5, 0.5])
        self.dt = dt
        self.world = w
        self.reset()
        self.world.render() # This visualizes the world we just constructed.
        # import pdb; pdb.set_trace()
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
        self.starting_region = np.random.choice(2)
        xs = np.linspace(np.pi/2, 2*np.pi+np.pi/2, 10)
        theta = np.random.choice(xs) % (np.pi*2)
        lane_markers_radius = self.inner_building_radius + (0.7) * (self.lane_marker_width - 2) #+ (0 + 0.5) * self.lane_marker_width
        dx = lane_markers_radius * np.cos(theta)
        dy = lane_markers_radius * np.sin(theta)
        if self.starting_region == 0:
            x = self.cb1.center.x + dx
            y = self.cb1.center.y + dy
        else:
            x = self.cb2.center.x + dx
            y = self.cb2.center.y + dy
        # x = self.world_width/2 + dx
        # y = self.world_height/2 + dy

        c1 = Car(Point(x, y), theta + np.pi/2)
        c1.max_speed = 30.0 # let's say the maximum is 30 m/s (108 km/h)
        c1.min_speed = 3.0 # let's say the maximum is 30 m/s (108 km/h)
        c1.velocity = Point(0.0, 0.0)
        self.agent = c1
        self.world.add(c1)
        if self.world.collision_exists(): # set the car smack in the center
            self.world.reset()
            c1 = Car(Point(self.center[0], self.center[1]), theta + np.pi/2)
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
    
    def get_info(self):
        return {}

    def label(self, state):
        signal, labels = {}, {}
        
        # import pdb; pdb.set_trace()
        for idx, distance in enumerate(self.distance_to_waypoints(state)):
            if distance <= 3:
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

    def distance_to_potholes(self, state):
        new_state = state * np.array([self.world_width, self.world_height, self.agent.max_speed, self.agent.max_speed, 2*np.pi]) 
        return np.array([np.linalg.norm([new_state[0] - wp.x, new_state[1] - wp.y]) for wp in self.potholes])

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
            #import pdb; pdb.set_trace()
            if save_dir is not None:
                self.world.visualizer.save_fig(save_dir)
                self.world.remove_agents()
                #self.world.visualizer.save_fig(save_dir)
    
            if save_states:
                np.save(save_dir + '.npy', np.array(states))
         
        # numpy_fig = mplfig_to_npimage(self.fig)  # convert it to a numpy array
        # return numpy_fig
    
    def between_circles_reward(self, relative_position):
        if abs(self.agent.x - self.center[0]) > (self.distance_between_circles / 2):
            return 0
        if abs(self.agent.y - self.center[1]) > (self.distance_between_circles / 2):
            return 0
        return np.linalg.norm(relative_position - self.relative_center) * 3
    
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
        #reward = (-1 * np.linalg.norm(u)) #* 0.25
        relative_position = np.array([self.agent.x/self.world_width, self.agent.y/self.world_height])
        # Reward is higher if the agent is closer to the edges of the map.
        circle_outer_reward = 2 * np.square(self.agent.x/self.world_width - 0.5) + 2 * np.square(self.agent.y/self.world_height - 0.5)
        reward_normalizing_constant = 40
        #reward = (min(self.distance_to_potholes(self.state)) / reward_normalizing_constant) ** 2
        # Reward is higher if the agent is closer to the center of the map.
        circle_inner_reward = np.square(min(abs(relative_position[0] - 1), relative_position[0])) + np.square(min(abs(relative_position[1] - 1), relative_position[1]))
        #reward = self.between_circles_reward(relative_position)
        #stay_centered_reward = (21.25 - np.linalg.norm(position - self.center)) / self.inner_building_radius  # negative penalty for distance from the center of the track
        terminated = self.world.collision_exists()
        self.state = self.get_state()
        # if (np.linalg.norm(self.state[2:4]*self.agent.max_speed)+1e-7) < self.agent.min_speed:
        #     import pdb; pdb.set_trace()
        # if self.distance_to_waypoints(self.state) < 2:

        return self.state, circle_outer_reward, terminated, {}
        
