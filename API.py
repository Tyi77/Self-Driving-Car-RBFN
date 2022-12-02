from tkinter import *
from tkinter import font
import math
import numpy as np
import random

import Main # Go to the right Entry Point.
import Train4D
import Train6D

'''
global variables: root, train4D_frame, train6D_frame
'''

class _font():
    @property
    def mjh34b(self):
        return font.Font(family='Microsoft JhengHei UI', size=34, weight='bold')
    @property
    def mjh20(self):
        return font.Font(family='Microsoft JhengHei UI', size=20)
    @property    
    def mjh14(self):
        return font.Font(family='Microsoft JhengHei UI', size=14)
    @property
    def mjh12(self):
        return font.Font(family='Microsoft JhengHei UI', size=12)

def create_window():
    global root
    root = Tk()
    root.title('HW2')
    # Designate height and width of the window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    app_width = 900
    app_height = 750

    x = (screen_width - app_width) // 2
    y = (screen_height - app_height) // 2

    root.geometry(f'{app_width}x{app_height}+{x}+{y}')

def create_frames():
    global train4D_frame, train6D_frame
    train4D_frame = Frame(root, width=900, height=750)
    train6D_frame = Frame(root, width=900, height=750)

def hide_all_frames():
    train4D_frame.pack_forget()
    train6D_frame.pack_forget()

def activate_train4D_frame():
    hide_all_frames()
    train4D_frame.pack(fill='both', expand=1)

def activate_train6D_frame():
    hide_all_frames()
    train6D_frame.pack(fill='both', expand=1)
    
def create_menu():
    main_menu = Menu(root)

    menu_options = Menu(main_menu, activebackground='gray', font=_font().mjh12, tearoff=0)

    main_menu.add_cascade(label='Options', menu=menu_options)
    menu_options.add_command(label='Train 4D', command=activate_train4D_frame)
    menu_options.add_separator()
    menu_options.add_command(label='Train 6D', command=activate_train6D_frame)
    menu_options.add_separator()
    menu_options.add_command(label='Exit', command=root.quit)

    root.config(menu=main_menu)

def set_frames():
    Train4D.set_frame_content()
    Train6D.set_frame_content()



def k_means(data, k, epsilon):
    '''
    k is the number of the clustering centers.\n
    Output: 'Centers' and 'Belongs'.
    '''
    # ---Check if k is a positive integer---
    if isinstance(k, float) and not k.is_integer():
        return None, None
    if k <= 0:
        return None, None
    # ---Initialize Centers---
    numbers = 0
    centers = random.sample(list(data), k)
    while True:
        centers = np.unique(centers, axis=0)
        numbers = len(centers)
        if numbers != k:
            tmp = random.sample(list(data), k - numbers)
            centers = np.concatenate((centers, tmp))
        else:
            break

    # ---Calculate---
    _limit_round = 100
    for r in range(_limit_round):
        # Initialize Belongs
        belongs = [] # All points to Each Center.
        for _ in range(k):
            belongs.append([])

        # Calculate Belongs
        for p in data:
            minDis = 1000
            target = -1
            sign = 0 ## Debug
            for idx_c, c in enumerate(centers):
                if np.linalg.norm(p - c, ord=2) <= minDis:
                    minDis = np.linalg.norm(p - c, ord=2)
                    target = idx_c
            # print(minDis)
            belongs[target].append(p)
                  
        # Adjust Centers
        count = 0
        for idx, b_list in enumerate(belongs):
            if b_list != []:
                meanPoint = np.mean(b_list, axis=0)
            else:
                print(f'The {idx}th b_list is empty.') ## DEBUG
                count += 1
                continue
            if np.linalg.norm(meanPoint - centers[idx]) < epsilon:
                count += 1

            centers[idx] = meanPoint
        if count == k:
            print(f'{r + 1} Round') ## Debug
            return centers, belongs
    print(print(f'{_limit_round} Round')) ## Debug
    return centers, belongs

def rbf_row(p, rbfArg):
    '''
    p: One element from the input array.\n
    rbfArg: The needed arguments. (Whole neurons in hidden layer)\n
    This function is used by np.vectorize()
    '''
    # print(np.array([math.exp( -1 * sigma * (np.linalg.norm(p - m) ** 2)) for (m, sigma) in rbfArg]))
    return np.array([math.exp( -1 * sigma * (np.linalg.norm(p - m) ** 2)) for (m, sigma) in rbfArg])

def rbfn_train(filePath, hidden_dim, identityC, sigma):
    '''
    rbfList: The array of RBFs which are expressed by (center, width). ps: two parameters
    weights: The array of weights.
    '''

    # ---Get Training Data---
    try:
        data = np.loadtxt(filePath)
    except:
        return
    # data = np.unique(data, axis=0) # remove identical data
    # np.random.shuffle(data)
    data_x, data_d = np.split(data, [-1], axis=1)
    data_d = data_d.flatten()

    # ---Initialize---
    # centers, belongs = k_means(data_x, hidden_dim, 0.01)
    # centers, belongs = art2(data_x, hidden_dim)
    # rbfList = [[c, np.mean([np.linalg.norm(c - b) for b in belongs[i]])] for i, c in enumerate(centers)] # element: [centerPosition, Width]
    # rbfList = [[c, sigma] for c in centers]

    # ---Initialize 2: full RBFN---
    rbfList = [[x, sigma] for x in data_x]

    # ---Training---
    rbf_row_func = np.vectorize(rbf_row, signature='(m),(n, 2)->(n)')
    # print(rbf_row_func)

    phi = rbf_row_func(data_x, rbfList) # shape: (data_num, hidden_dim)
    # phi = np.insert(phi, 0, 1, axis=1)
    # print(phi.shape) ## DEBUG
    # phi_sudoInverse = np.linalg.inv(phi.T @ phi + 1e-11 * np.eye(phi.shape[0])) @ phi.T
    phi_sudoInverse = np.linalg.pinv(phi)
    # phi_sudoInverse = np.linalg.inv(phi)

    # ---Calculate loss---
    # print(phi_sudoInverse @ phi)


    weights = phi_sudoInverse @ data_d # shape: (hidden_dim+-, )

    return rbfList, weights



class Line:
    ...

class Line():
    def __init__(self, *arg):
        if len(arg) == 2:
            self.p1 = arg[0]
            self.p2 = arg[1]
        else:
            self.p1 = [arg[0], arg[1]]
            self.p2 = [arg[2], arg[3]]
    
    @property
    def length(self):
        return math.sqrt((self.p1[0] - self.p2[0]) ** 2 + (self.p1[1] - self.p2[1]) ** 2)
    
    @property
    def endpoints(self):
        return [self.p1, self.p2]
    
    def unitIntersectLength(self, line2: Line):
        '''
        Input:
            line2: Line
                The unit line.
        Output:
            t: float
                Find the vector between the intersect point from the origin point.
                Then calculate the length of that vector. 
        '''
        # Fetch the origin point of two lines
        # Fetch the vectors
        x0, y0 = line2.p1[0], line2.p1[1] # car point
        a, b = line2.p2[0] - line2.p1[0], line2.p2[1] - line2.p1[1] # car vector
        x1, y1 = self.p1[0], self.p1[1] # line first endpoint
        c, d = self.p2[0] - self.p1[0], self.p2[1] - self.p1[1] # line vector
        
        # Calculate the length
        try:
            t = (d * (x1 - x0) - c * (y1 - y0)) / (d * a - c * b) # Distance
            if t < 0: # Must be at the front side of the vector
                return None

            s = (b * (x1 - x0) - a * (y1 - y0)) / (d * a - c * b)
            if 0 <= s <= 1: # The intersecion point must be on the self line
                return t
            else:
                return None

        except ZeroDivisionError as e: # Parallel or Coinciding line
            return None

if __name__ == '__main__':
    Main.start_up()