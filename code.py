# Mondrian Generator

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

%matplotlib inline

def dimensions(box, size=None):
    if box is None:
        if size is not None:
            return np.zeros(size)
        else: 
            raise ValueError('If box is not provided, you must specify size.')
            
    return np.diff(box, axis=1).flatten()

def linear_dimension(box):
    if box is None:
        return 0
    return dimensions(box).sum()

def interval_difference(outer_interval, inner_interval):
    lower_outer, upper_outer = outer_interval
    lower_inner, upper_inner = inner_interval
    return [lower_outer, lower_inner], [upper_inner, upper_outer]

def sample_interval_difference(outer_interval, inner_interval):
    intervals = interval_difference(outer_interval, inner_interval)
    dimensions = [np.diff(intervals[0])[0], np.diff(intervals[1])[0]]
    chosen_interval_index = np.random.choice(range(len(intervals)), p=dimensions/np.sum(dimensions))
    chosen_interval = intervals[chosen_interval_index]
    return np.random.uniform(low=chosen_interval[0], high=chosen_interval[1], size=1)[0], chosen_interval_index

def random_axis(dimensions):
    return np.random.choice(range(len(dimensions)), 
                            p=dimensions/np.sum(dimensions))

def random_cut(box, axis):
    return np.random.uniform(low=self.box[axis][0], 
                             high=self.box[axis][1], 
                             size=1)[0]

def cost_next_cut(linear_dimension):
    return np.random.exponential(scale=1.0/linear_dimension, size=1)[0]

def new_cut_proposal(self):
    cost = self.cost_next_cut()
    axis = self.random_axis()
    cut_point = self.random_cut(axis)
    return cost, axis, cut_point

def cut_boxes(box, cut_axis, cut_point):
    left = box.copy()
    right = box.copy()
    low, high = box[cut_axis]

    if cut_point <= low or cut_point >= high:
        raise ValueError('Point is not in interval.')

    left[cut_axis] = [low, cut_point]
    right[cut_axis] = [cut_point, high]
    return left, right
  
  class Mondrian(object):
    def __init__(self, box, budget):
        self.box = box
        self.budget = budget
        self.cut_point = None
        self.cut_axis = None
        self.cut_budget = None
        self.left = None
        self.right = None
        
    def extended_by(self, box):
        if self.box is None:
            return True
        return all((box[:, 0] <= self.box[:, 0]) & (box[:, 1] >= self.box[:, 1]))
    
    def contains(self, point):
        if self.box is None:
            return False
        return all(box[:,0] <= point) & (box[:,1] >= point)
    
    def has_cut(self):
        return self.cut_axis is not None
    
    def is_empty(self):
        return self.box is None
    
def grow_mondrian(box, budget, given_mondrian=None):
    if given_mondrian is None:
        given_mondrian = Mondrian(None, budget)

    if not given_mondrian.extended_by(box):
        raise ValueError('Incompatible boxes: given mondrian box must be contained in new box.')
    
    mondrian = Mondrian(box, budget)
    
    cost = cost_next_cut(linear_dimension(box) - linear_dimension(given_mondrian.box))
    
    next_budget = budget - cost
    
    given_mondrian_next_budget = given_mondrian.cut_budget if given_mondrian.has_cut() else 0
    
    
    if next_budget < given_mondrian_next_budget:        
        if given_mondrian.has_cut():
            mondrian.cut_axis = given_mondrian.cut_axis
            mondrian.cut_point = given_mondrian.cut_point
            mondrian.cut_budget = given_mondrian_next_budget

            left, right = cut_boxes(box, mondrian.cut_axis, mondrian.cut_point)

            mondrian.left = grow_mondrian(left, given_mondrian_next_budget, given_mondrian.left)
            mondrian.right = grow_mondrian(right, given_mondrian_next_budget, given_mondrian.right)
    else:
        dimensions_outer = dimensions(box)
        dimensions_inner = dimensions(given_mondrian.box, size=len(box))
        mondrian.cut_axis = random_axis(dimensions_outer - dimensions_inner)
        outer_interval = box[mondrian.cut_axis]
        
        if given_mondrian.is_empty():
            inner_interval = [outer_interval[0], outer_interval[0]]
        else:
            inner_interval = given_mondrian.box[mondrian.cut_axis]

        mondrian.cut_point, cut_side = sample_interval_difference(outer_interval, inner_interval)
        mondrian.cut_budget = next_budget
        
        left, right = cut_boxes(box, mondrian.cut_axis, mondrian.cut_point)

        if cut_side: # entire given_mondrian to the left
            mondrian.left = grow_mondrian(left, next_budget, given_mondrian)
            mondrian.right = grow_mondrian(right, next_budget, Mondrian(None, next_budget))
        else: # all given_mondrian to the right
            mondrian.left = grow_mondrian(left, next_budget, Mondrian(None, next_budget))
            mondrian.right = grow_mondrian(right, next_budget, given_mondrian)
            
    return mondrian
  
  def get_random_color():
    return np.random.choice(['blue', 'red', 'yellow', 'white'], 1)[0]
    
def box_2d(box, color=None, alpha=None):
    low_x, high_x = box[0]
    low_y, high_y = box[1]
    width = high_x - low_x
    height = high_y - low_y
    
    if color is None:
        color = 'white'
    if alpha is None:
        alpha = 1
        
    lower_left_corner = np.array([low_x, low_y])
    
    return mpatches.Rectangle(lower_left_corner, width, height, 
                              color=color, ec='black', linewidth=2, 
                              alpha=alpha)

def boxes(m, box_collection, color=None):
    random_color = False
    
    if color == 'true_mondrian':
        random_color = True
        color = get_random_color()
    
    box_collection.append(box_2d(m.box, color))
    
    if m.has_cut():
        color = 'true_mondrian' if random_color else color
        boxes(m.left, box_collection, color)
        boxes(m.right, box_collection, color)
        
def plot_coloured_mondrian(m, ax, color=None, given_mondrian=None):
    if given_mondrian is None:
        given_mondrian = Mondrian(None, budget)
        
    box_collection = []
    boxes(m, box_collection, color)
    
    if not given_mondrian.is_empty():
        box_collection.append(box_2d(given_mondrian.box, 
                                     color='black', alpha=0.1))
        
    collection = PatchCollection(box_collection, match_original=True)
    ax.add_collection(collection)
    
    ax.axis('off')
    ax.autoscale()
    #plt.show()
    
def random_mondrians(box, budget, given_mondrian=None, color=None, rows=1, columns=1, figsize=(15, 15)):
    if given_mondrian is None:
        given_mondrian = Mondrian(None, budget)
        
    if rows == 1 and columns == 1:
        fig, ax = plt.subplots(figsize=figsize)  
        plot_coloured_mondrian(grow_mondrian(box, budget, given_mondrian), 
                               ax, color=color, given_mondrian=given_mondrian)
    else:
        fig, ax = plt.subplots(rows, columns, figsize=figsize)  

        for row in range(rows):
            for col in range(columns):
                if not given_mondrian.is_empty() and row == 0 and col ==0:
                    plot_coloured_mondrian(given_mondrian, ax[row, col], color=color)
                else:
                    plot_coloured_mondrian(grow_mondrian(box, budget, given_mondrian), 
                                           ax[row, col], color=color, given_mondrian=given_mondrian)
                    
def growing_mondrians(initial_box, budget, rows=1, columns=1, figsize=(15, 15)):
    if rows == 1 and columns == 1:
        fig, ax = plt.subplots(figsize=figsize)  
        plot_coloured_mondrian(grow_mondrian(box, budget), 
                               ax, color=None)
    else:
        fig, ax = plt.subplots(rows, columns, figsize=figsize)  
        
        given_mondrian = Mondrian(None, budget)
        box = initial_box
        for row in range(rows):
            for col in range(columns):
                mondrian = grow_mondrian(box, budget, given_mondrian)
                plot_coloured_mondrian(mondrian, ax[row, col], color=None, given_mondrian=given_mondrian)
                given_mondrian = mondrian
                box = 2 * box
                
box = np.array([[0.0, 1.0], [0.0, 1.0]])

random_mondrians(box, 1, rows=3, columns=3, color='true_mondrian')
plt.savefig("MondrianExmpl.pdf")
