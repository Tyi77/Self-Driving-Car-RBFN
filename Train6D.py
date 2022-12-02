import numpy as np
import math
from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

import API
import Main # Go to the right Entry Point.

def car_cal(wheelAngle, x1, y1, o1, carSize):
    '''
    Input:
        wheelAngle: float
            The turned angle of the wheel.
        x1, y1: float, float
            The positon of the car.
        o1: float
            The orientation of the car.
    Output:
        newX, newY: float, float
            The new position of the car.
        orientation: float
            The new orientation of the car.
    '''
    _b = carSize # The radius of the car
    _x1 = x1
    _y1 = y1
    _o1 = o1 / 180 * math.pi
    _theta = wheelAngle / 180 * math.pi

    newX = _x1 + math.cos(_o1 + _theta) + math.sin(_theta) * math.sin(_o1)
    newY = _y1 + math.sin(_o1 + _theta) - math.sin(_theta) * math.cos(_o1)
    orientation = _o1 - math.asin(2 * math.sin(_theta) / _b)
    orientation = orientation / math.pi * 180

    if orientation < -90:
        orientation = -90
    elif 270 < orientation:
        orientation = 270

    return newX, newY, orientation

def detect_dist(s, space):
    '''
    Input: s[x, y, o], space
    Output: forward, right45, left45
    '''
    # ---Create detecting unit lines---
    _o = s[2] / 180 * math.pi # Transform degree to radian
    _or45 = (s[2] - 45) / 180 * math.pi
    _ol45 = (s[2] + 45) / 180 * math.pi

    forward_line = API.Line(s[0], s[1], s[0] + math.cos(_o), s[1] + math.sin(_o))
    right45_line = API.Line(s[0], s[1], s[0] + math.cos(_or45), s[1] + math.sin(_or45))
    left45_line = API.Line(s[0], s[1], s[0] + math.cos(_ol45), s[1] + math.sin(_ol45))

    # ---Calculate distance---
    forward_min, right45_min, left45_min = 50, 50, 50
    for line in space:
        forward_dist = line.unitIntersectLength(forward_line)
        right45_dist = line.unitIntersectLength(right45_line)
        left45_dist = line.unitIntersectLength(left45_line)

        if forward_dist is not None:
            if forward_dist < forward_min:
                forward_min = forward_dist
        if right45_dist is not None:
            if right45_dist < right45_min:
                right45_min = right45_dist
        if left45_dist is not None:
            if left45_dist < left45_min:
                left45_min = left45_dist
    
    return forward_min, right45_min, left45_min

def get_trace(fArea, rSpace, rbfs, w):
    # ---Create the buffer to store the result---
    result = [[0, 0, 22., 8.4853, 8.4853, 0, 90]] # [x, y, forward_dist, right45_dist, left45_dist, theta, orientation]

    # ---Configuration---
    rbf_row_func = np.vectorize(API.rbf_row, signature='(m),(n, 2)->(n)')
    status_start = [0, 0, 90] # [x, y, carAngle]
    dist_start = [0., 0., 22., 8.4853, 8.4853] # [x, y, forward_dist, right45_dist, left45_dist]
    b = 3 # car size

    # ---Build the road Space---
    space = []
    for idx in range(len(rSpace) - 1):
        space.append(API.Line(rSpace[idx], rSpace[idx+1]))

    # ---Run the car---
    status = status_start
    dist = dist_start
    theta = 0 # The angle of the car itself
    while True:
        # Fetch the theta
        # theta = np.matmul(np.concatenate(([1], rbf_row_func(dist, rbfs))), w)
        theta = np.matmul(rbf_row_func(dist, rbfs), w)
        # print(theta, end=' ')
        if theta > 40:
            theta = 40.
        elif theta < -40:
            theta = -40.
        
        # Change the position and orientation of the car
        status = car_cal(theta, status[0], status[1], status[2], b)
        
        ### SKIP
        # carArea = [[status[0] - b, status[0] + b], [status[1] - b, status[1] + b]] # [x_left, x_right], [y_backward, y_forward]

        # # Check whether the car is in the road.
        # if -6 <= carArea[0][0] and carArea[0][1] <= 6 and -3 <= carArea[1][0] and carArea[1][1] <= 22:
        #     pass
        # elif 6 <= carArea[0][0] and carArea[0][1] <= 18 and 10 <= carArea[1][0] and carArea[1][1] <= 22:
        #     pass
        # elif 18 <= carArea[0][0] and carArea[0][1] <= 30 and 10 <= carArea[1][0] and carArea[1][1] <= 50:
        #     pass
        # else:
        #     break
        ###

        # Change dist
        dist = np.concatenate(((status[0], status[1]), detect_dist(status, space)))

        tmp = np.concatenate((dist, [theta], [status[2]]))
        if tmp[2] >= b and tmp[3] >= b and tmp[4] >= b:
            result.append(tmp)
        else:
            break

        # Check whether the car went through the finish line.
        if status[1] > fArea[1][1]:
            return result

    return result

def plot_trace(*args):
    # ---Fetch Args---
    try:
        # _hidden_dim = int(args[0].get())
        # _hidden_dim = 100
        # _identity_coefficient = float(args[0].get())
        # _sigma = float(args[1].get())
        # _tkcanvas, _fig = args[2], args[3]

        
        _hidden_dim = 500
        _identity_coefficient = 0.0001
        _sigma = float(args[0].get())
        _tkcanvas, _fig = args[1], args[2]
        _forward_dist_text, _right45_dist_text, _left45_dist_text = args[3], args[4], args[5]
    except:
        return


    # ---Get the Trained Model---
    rbfList, weights = API.rbfn_train('train6DAll.txt', _hidden_dim, _identity_coefficient, _sigma)
    
    # ---Get the Road Points---
    startLine_x = [-6, 6]
    startLine_y = [0, 0]
    finishLine, roadSpace = [], []
    with open('軌道座標點.txt', mode='r') as file:
        for idx, line in enumerate(file):
            if idx == 0:
                pass
            elif idx == 1 or idx == 2:
                finishLine.append(list(map(int, line.strip().split(','))))
            else:
                roadSpace.append(list(map(int, line.strip().split(','))))
    finishArea = [
        [finishLine[0][0], finishLine[0][1]],
        [finishLine[0][0], finishLine[1][1]],
        [finishLine[1][0], finishLine[1][1]],
        [finishLine[1][0], finishLine[0][1]],
        [finishLine[0][0], finishLine[0][1]]
    ]
    finishArea_x = [p[0] for p in finishArea]
    finishArea_y = [p[1] for p in finishArea]
    roadSpace_x = [p[0] for p in roadSpace]
    roadSpace_y = [p[1] for p in roadSpace]

    # ---Get the Trace---
    trace = get_trace(finishLine, roadSpace, rbfList, weights) # element: [x, y, o]

    # ---Ouput the Trace---
    output = [[p[0], p[1], p[2], p[3], p[4], p[5]] for p in trace]
    np.savetxt('carTrace6D.txt', output, fmt='%06.4f')

    # ---Plot---
    _fig.clear()
    ax = _fig.add_subplot(111)

    def init():
        ax.clear()
        ax.plot(startLine_x, startLine_y, color='k')
        ax.plot(roadSpace_x, roadSpace_y, color='k')
        ax.plot(finishArea_x, finishArea_y, color='r')
        ax.axis('tight')
    
    def run(data):
        circle = plt.Circle((data[0], data[1]), 3, color='k', fill=False)
        x = data[0] + 3 * math.cos(data[6] / 180 * math.pi)
        y = data[1] + 3 * math.sin(data[6] / 180 * math.pi)

        ax.scatter(data[0], data[1], color='b')
        ax.plot([data[0], x], [data[1], y], color='r')
        ax.add_patch(circle)
    
        _forward_dist_text.config(text=str(round(data[2], 4)))
        _right45_dist_text.config(text=str(round(data[3], 4)))
        _left45_dist_text.config(text=str(round(data[4], 4)))

    ani = animation.FuncAnimation(_fig, run, init_func = init, frames=trace, interval=100, repeat=False)
    ani.save('carTrace.gif')
    ani.__del__()

    _tkcanvas.draw()

# def clear(arg):
#     arg.clf()

def set_frame_content():
    # ---Fetch globals properties from API.py---
    # root = API.root
    train6D_frame = API.train6D_frame
    _font = API._font()

    # ---Widgets Setting---
    title = Label(train6D_frame, text='Train 6D', font=_font.mjh34b)

    playground_fig = Figure(figsize=(10, 10))
    playground_canvas = FigureCanvasTkAgg(playground_fig, master=train6D_frame)

    forward_dist_lb = Label(train6D_frame, text='Forward Dist', font=_font.mjh12)
    forward_dist = Label(train6D_frame, text='', font=_font.mjh12)
    right45_dist_lb = Label(train6D_frame, text='Right45 Dist', font=_font.mjh12)
    right45_dist = Label(train6D_frame, text='', font=_font.mjh12)
    left45_dist_lb = Label(train6D_frame, text='Left45 Dist', font=_font.mjh12)
    left45_dist = Label(train6D_frame, text='', font=_font.mjh12)

    argument_area = LabelFrame(train6D_frame, relief='ridge', borderwidth=2)

    # hidden_dim_lb = Label(train6D_frame, text='Hidden\nDimension', font=_font.mjh12)
    # hidden_dim = Entry(train6D_frame, font=_font.mjh12)

    # identityC_lb = Label(train6D_frame, text='Identity\nCoefficient', font=_font.mjh12)
    # identityC = Spinbox(train6D_frame, from_=0.0000001, to= 0.1, increment=0.0000001, font=_font.mjh12)

    sigma_lb = Label(train6D_frame, text='Sigma', font=_font.mjh12)
    # sigma = Scale(train6D_frame, from_=0.0, to=0.0001, resolution=0.000001, tickinterval=0.5, orient='horizontal', cursor='cross')
    # sigma = Entry(train6D_frame, font=_font.mjh12)
    sigma = Spinbox(train6D_frame, from_=0.000000001, to= 0.1, increment=0.00000001, font=_font.mjh12)

    # args = [hidden_dim, sigma, playground_canvas, playground_fig]
    # args = [identityC, sigma, playground_canvas, playground_fig]
    args = [sigma, playground_canvas, playground_fig, forward_dist, right45_dist, left45_dist]
    submit_bt = Button(train6D_frame, text='Run!', font=_font.mjh12, command=lambda: plot_trace(*args))

    # clear_bt = Button(train6D_frame, text='Clear!', font=_font.mjh12, command=lambda: clear(playground_fig))

    # ---Place widgets---
    title.place(relx=0.5, rely=0, relwidth=0.9, relheight=0.1, anchor=N)
    
    playground_canvas.get_tk_widget().place(relx=0.5, rely=0.1, relwidth=0.6, relheight=0.6, anchor=N)
    
    forward_dist_lb.place(relx=0.9, rely=0.35, relwidth=0.15, relheight=0.04, anchor=N)
    forward_dist.place(relx=0.9, rely=0.4, relwidth=0.15, relheight=0.04, anchor=N)
    right45_dist_lb.place(relx=0.9, rely=0.45, relwidth=0.15, relheight=0.04, anchor=N)
    right45_dist.place(relx=0.9, rely=0.5, relwidth=0.15, relheight=0.04, anchor=N)
    left45_dist_lb.place(relx=0.9, rely=0.55, relwidth=0.15, relheight=0.04, anchor=N)
    left45_dist.place(relx=0.9, rely=0.6, relwidth=0.15, relheight=0.04, anchor=N)

    argument_area.place(relx=0.5, rely=0.8, relwidth=0.6, relheight=0.125, anchor=CENTER)

    # hidden_dim_lb.place(relx=0.3, rely=0.8, relwidth=0.15, height=40, anchor=CENTER)
    # hidden_dim.place(relx=0.4, rely=0.8, relwidth=0.1, height=40, anchor=CENTER)
    # identityC_lb.place(relx=0.3, rely=0.8, relwidth=0.15, height=40, anchor=CENTER)
    # identityC.place(relx=0.4, rely=0.8, relwidth=0.1, height=40, anchor=CENTER)

    # sigma_lb.place(relx=0.5, rely=0.8, relwidth=0.05, height=40, anchor=CENTER)
    # sigma.place(relx=0.61, rely=0.8, relwidth=0.13, height=40, anchor=CENTER)
    sigma_lb.place(relx=0.38, rely=0.8, relwidth=0.06, height=40, anchor=CENTER)
    sigma.place(relx=0.5, rely=0.8, relwidth=0.13, height=40, anchor=CENTER)

    submit_bt.place(relx=0.65, rely=0.8, width=80, height=40, anchor=CENTER)
    
    # clear_bt.place(relx=0.75, rely=0.8, width=80, height=40, anchor=CENTER)

if __name__ == '__main__':
    Main.start_up()