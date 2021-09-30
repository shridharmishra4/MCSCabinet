import copy
import sys
import cv2
import tkinter as tk
from tkinter import Canvas, filedialog
from PIL import Image, ImageTk
from collections import defaultdict
import argparse
import pickle
import numpy as np
from copy import deepcopy
import os


class Application:
    """
    { {<frame_number>: {'left': {<grid_id>: {'coordinates': [0, 0, 100, 135], 'clicked': True},
                                 <grid_id>: {'coordinates': [100, 0, 200, 135], 'clicked': False},
                                 },
                        'right': {<grid_id>: {'coordinates': [0, 0, 100, 135], 'clicked': True},
                                  <grid_id>: {'coordinates': [100, 0, 200, 135], 'clicked': False},
                                  }
                        }
    }

    
    """
    def __init__(self, filename,output_path = "./"):
        self.window = tk.Tk()

        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        filename = filedialog.askopenfilename()
        output_path_heatmap = os.path.dirname(filename)
        interaction_path = filename.split(".")[0]+".pkl"
        self.interaction_history = pickle.load(open(interaction_path, "rb"))
        self.interaction_history_shape = self.interaction_history["shape"]
        print(self.interaction_history)
        filename_items = filename.split("/")
        right_name = filename_items[-1].split("_")
        M_filename =os.path.join(output_path_heatmap, "_".join(right_name[:-1]) + "_M.p" ) 
        right_name[-1] = "right.mp4"
        right_name = "_".join(right_name)
        filename_items[-2],filename_items[-1] = "right",right_name
        front_view = "/".join(filename_items)
        # front_view = os.path.join("/".join(video_path.split("/")[:-1]),f"right/{video_name}")

        self.vs = cv2.VideoCapture(front_view) # capture video frames, 0 is your default video camera
        ok, frame = self.vs.read() # get first frame
        self.output_path = output_path  # store output path
        self.current_image = None  # current image from the camera
        
        isDynamic = True
        self.prev_images = []  # Prev image from the camera
        self.prev_map = []
        self.M = self.load_maxtrix(M_filename)
        self.aspect_ratio = (180, 270)
        img = self.wrap_perspective(frame)
        if isDynamic:
            img = self.wrap_perspective(frame)
        else:
            img = self.img = cv2.imread("door.jpg")
        # cv2.imshow("img",img)
        # cv2.waitKey(0)
        cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
        current_image = Image.fromarray(cv2image)  # convert image for PIL

        self.img = ImageTk.PhotoImage(image=current_image)
        self.left_cell_state = defaultdict(dict)
        self.right_cell_state = defaultdict(dict)

        self.temp_images = []
        self.left_rect_dict = {}
        self.right_rect_dict = {}
        self.global_cell_states = defaultdict(dict)
        self.current_frame = 0
        self.n = 5
        
        self.width, self.height = self.img.width(), self.img.height()

        self.pad = 0

        self.xsize = (self.width) / self.n 
        self.ysize = (self.height) / self.n

        c_width, c_height = self.img.width(), self.img.height()

        self.window.title("Correction module")  # set window title
        # self.destructor function gets fired when the window is closed
        self.window.protocol('WM_DELETE_WINDOW', self.destructor)

        self.images_frame = tk.Frame(self.window,width=1080, height=720)
        self.images_frame.pack(side="top")

        self.left_imageFrame = tk.Canvas(self.images_frame, width=c_width, height=c_height)
        self.left_imageFrame.addtag_all("left")
        # self.left_imageFrame.pack(side="left", expand=True)
        self.left_img = self.left_imageFrame.create_image(0,0,image=self.img, anchor="nw")
        self.left_imageFrame.pack(side="left")

        # self.label_frame = tk.Label(self.images_frame, text=str(self.current_frame))
        # Create a label which displays the camera feed
        self.center_imageFrame = tk.Frame(self.images_frame,width=512, height=360)
        self.center_imageFrame.pack(side="left")

        self.right_imageFrame = tk.Canvas(self.images_frame, width=c_width, height=c_height)
        self.right_img = self.right_imageFrame.create_image(0,0,image=self.img, anchor="nw")
        self.right_imageFrame.pack(side="left")
        self.right_imageFrame.addtag_all("right")

        # self.left_panel = tk.Label(self.left_imageFrame)
        # self.left_panel.pack(side="bottom",expand=True)
        # self.left_panel.bind("<Button 1>",self.printcoords)
        self.left_imageFrame.bind('<ButtonPress-1>', self.left_click_callback)
        self.right_imageFrame.bind('<ButtonPress-1>', self.right_click_callback)

 
        self.center_panel = tk.Label(self.center_imageFrame) 
        self.center_panel.pack(side="left",expand=True)

        # self.right_panel = tk.Label(self.right_imageFrame)
        # self.right_panel.pack(side="left",expand=True)
        # self.right_panel.bind("<Button 2>",self.printcoords)

        self.prev_next = tk.Frame(self.window)
        self.prev_next.pack(side="bottom")
        
        # create a button, that when pressed, will take the current frame and save it to file
        prev = tk.Button(self.prev_next,text="Previous Frame", command=self.prev_frame)
        # prev.grid(row=1, column=0,sticky =tk.N)
        prev.pack(side=tk.LEFT)
        
        next = tk.Button(self.prev_next,text="Next Frame", command=self.next_frame)
        # next.grid(row=1, column=1,sticky = tk.N)
        next.pack( side=tk.LEFT)

        self.window.bind('<d>', self.next_frame)
        self.window.bind('<a>', self.prev_frame)
        self.window.bind('<c>', self.copy_prev_frame_state)

        self.next_frame()
        

        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
    

        #mouseclick event
    
    def load_maxtrix(self,filename):
        self.M = pickle.load(open(filename, "rb"))
        return self.M
        print("[INFO] loaded matrix from file: {}".format("model.p"))
    
    def wrap_perspective(self,img):
        """ Apply the perspective transformation matrix to the image """

        if self.M is not None:
            return cv2.warpPerspective(img, self.M, (self.aspect_ratio[0], self.aspect_ratio[1]))
    
    def draw_grid(self,canvas):
        self.cells = []
        state = self.interaction_history.get(self.current_frame)


        for iy in range(self.n):
            for ix in range(self.n):
                i = iy*self.n+ix
                xpad, ypad = self.pad * (ix+1), self.pad * (iy+1) 
                x, y = xpad + ix*self.xsize, ypad + iy*self.ysize
                # rect = canvas.create_rectangle(int(x), int(y), int(x+self.xsize),
                #                            int(y+self.ysize), fill="")
                self.left_cell_state[i]["coordinates"] = [int(x), int(y), int(x+self.xsize),
                                           int(y+self.ysize)]
                self.left_cell_state[i]["clicked"] = False
                self.left_cell_state[i]["clicked"] = False

                self.right_cell_state[i]["coordinates"] = [int(x), int(y), int(x+self.xsize),
                                           int(y+self.ysize)]
                self.right_cell_state[i]["clicked"] = False

                rect = self.create_translucent_rectangle(canvas,int(x), int(y), int(x+self.xsize),
                            int(y+self.ysize), fill="")
                self.cells.append(rect)
    
    def draw_grid_left(self):
        canvas=self.left_imageFrame
        self.cells = []
        left_state = None
        i_interaction = None
        rect_dect = {}
        state = self.interaction_history.get(self.current_frame)
        if state is not None:
            left_state = state.get("left")
            if left_state is not None:
                x = (left_state[0]/self.interaction_history_shape[0])*self.width
                y = (left_state[1]/self.interaction_history_shape[1])*self.height
                # x = (self.interaction_history_shape[0]-left_state[0]/self.interaction_history_shape[0])*self.width
                # y = (self.interaction_history_shape[1]-left_state[1]/self.interaction_history_shape[1])*self.height
                # Did the user click a cell in the grid?
                # Indexes into the grid of cells (including padding)
                ix = int(x // (self.xsize + self.pad))
                iy = int(y // (self.ysize + self.pad))
                xc = x - ix*(self.xsize + self.pad) - self.pad
                yc = y - iy*(self.ysize + self.pad) - self.pad
                if ix < self.n and iy < self.n and 0 < xc < self.xsize and 0 < yc < self.ysize:
                    i_interaction = iy*self.n+ix

            
        for iy in range(self.n):
            for ix in range(self.n):
                i = iy*self.n+ix
                xpad, ypad = self.pad * (ix+1), self.pad * (iy+1) 
                x, y = xpad + ix*self.xsize, ypad + iy*self.ysize
                # rect = canvas.create_rectangle(int(x), int(y), int(x+self.xsize),
                #                            int(y+self.ysize), fill="")
                self.left_cell_state[i]["coordinates"] = [int(x), int(y), int(x+self.xsize),int(y+self.ysize)]

                if i==i_interaction:
                    # print(i_interaction)
                    print(x,y,self.width,self.height)
                    self.left_cell_state[i]["clicked"] = True
                    rect = self.create_translucent_rectangle(canvas,int(x), int(y), int(x+self.xsize),
                                                            int(y+self.ysize),fill="red",alpha=0.5)
                else:
                    self.left_cell_state[i]["clicked"] = False
                    rect = self.create_translucent_rectangle(canvas,int(x), int(y), int(x+self.xsize),
                                int(y+self.ysize), fill="")
                self.left_cell_state[i]["rectangle"] = rect
                

        cpy = deepcopy(self.left_cell_state)
        self.global_cell_states[self.current_frame]["left"] = cpy

    def draw_grid_right(self):
        canvas=self.right_imageFrame
        self.cells = []
        right_state = None
        i_interaction = None
        state = self.interaction_history.get(self.current_frame)
        # state = self.global_cell_states[self.current_frame]["right"]
        if state is not None:
            right_state = state.get("right")
            if right_state is not None:
                x = (right_state[0]/self.interaction_history_shape[0])*self.width
                y = (right_state[1]/self.interaction_history_shape[1])*self.height
                # x = (self.interaction_history_shape[0]-right_state[0]/self.interaction_history_shape[0])*self.width
                # y = (self.interaction_history_shape[1]-right_state[1]/self.interaction_history_shape[1])*self.height
                # Did the user click a cell in the grid?
                # Indexes into the grid of cells (including padding)
                ix = int(x // (self.xsize + self.pad))
                iy = int(y // (self.ysize + self.pad))
                xc = x - ix*(self.xsize + self.pad) - self.pad
                yc = y - iy*(self.ysize + self.pad) - self.pad
                if ix < self.n and iy < self.n and 0 < xc < self.xsize and 0 < yc < self.ysize:
                    i_interaction = iy*self.n+ix

            
        for iy in range(self.n):
            for ix in range(self.n):
                i = iy*self.n+ix
                xpad, ypad = self.pad * (ix+1), self.pad * (iy+1) 
                x, y = xpad + ix*self.xsize, ypad + iy*self.ysize
                # rect = canvas.create_rectangle(int(x), int(y), int(x+self.xsize),
                #                            int(y+self.ysize), fill="")
                self.right_cell_state[i]["coordinates"] = [int(x), int(y), int(x+self.xsize),int(y+self.ysize)]

                if i==i_interaction:
                    # print(i_interaction)
                    self.right_cell_state[i]["clicked"] = True
                    rect = self.create_translucent_rectangle(canvas,int(x), int(y), int(x+self.xsize),
                                                            int(y+self.ysize),fill="red",alpha=0.5)
                else:
                    self.right_cell_state[i]["clicked"] = False
                    rect = self.create_translucent_rectangle(canvas,int(x), int(y), int(x+self.xsize),
                                int(y+self.ysize), fill="")
                self.right_cell_state[i]["rectangle"] = rect

        cpy = deepcopy(self.right_cell_state)
        self.global_cell_states[self.current_frame]["right"] = cpy   

    def create_translucent_rectangle(self,canvas, x1, y1, x2, y2, **kwargs):

        """Create a rectangle with coordinates (x1, y1) and (x2, y2)"""

        if 'alpha' in kwargs:
            alpha = int(kwargs.pop('alpha') * 255)
            fill = kwargs.pop('fill')
            fill = canvas.winfo_rgb(fill) + (alpha,)
            image = Image.new('RGBA', (x2-x1, y2-y1), fill)
            self.temp_images.append(ImageTk.PhotoImage(image))
            # canvas.imgtk = image
            return canvas.create_image(x1, y1, image=self.temp_images[-1], anchor=tk.NW)
        return canvas.create_rectangle(x1, y1, x2, y2, **kwargs)




    def left_click_callback(self,event):
        
        """Function called when someone clicks on the grid canvas."""
        # print("clicked at", event.x, event.y)
        x, y = event.x, event.y
        self.left_cell_state = deepcopy(self.global_cell_states[self.current_frame]["left"])
        
        # Did the user click a cell in the grid?
        # Indexes into the grid of cells (including padding)
        ix = int(x // (self.xsize + self.pad))
        iy = int(y // (self.ysize + self.pad))
        xc = x - ix*(self.xsize + self.pad) - self.pad
        yc = y - iy*(self.ysize + self.pad) - self.pad
        if ix < self.n and iy < self.n and 0 < xc < self.xsize and 0 < yc < self.ysize:
            i = iy*self.n+ix
            # self.left_imageFrame.itemconfig(self.cells[i], fill="blue",stipple="gray50")
            coord = self.left_cell_state[i]["coordinates"]

            if not self.left_cell_state[i]["clicked"]:
                self.left_cell_state[i]["clicked"] = True
                im = self.create_translucent_rectangle(self.left_imageFrame,coord[0],
                                                coord[1], coord[2], coord[3], 
                                                fill="red",alpha=0.5)
                self.left_cell_state[i]["rectangle"] = im
            else:
                print(self.left_cell_state[i])
                self.left_cell_state[i]["clicked"] = False
                self.left_imageFrame.delete(self.left_cell_state[i]["rectangle"])
                # print(self.left_cell_state[i])
            # print(self.current_frame,self.left_cell_state[i]["clicked"])
        cpy = deepcopy(self.left_cell_state)
        self.global_cell_states[self.current_frame]["left"] = cpy
            # print(self.global_cell_states[self.current_frame]["left"])
    
    def right_click_callback(self,event):
        """Function called when someone clicks on the grid canvas."""
        x, y = event.x, event.y
        self.right_cell_state = deepcopy(self.global_cell_states[self.current_frame]["right"])
        
        # Did the user click a cell in the grid?
        # Indexes into the grid of cells (including padding)
        ix = int(x // (self.xsize + self.pad))
        iy = int(y // (self.ysize + self.pad))
        xc = x - ix*(self.xsize + self.pad) - self.pad
        yc = y - iy*(self.ysize + self.pad) - self.pad
        if ix < self.n and iy < self.n and 0 < xc < self.xsize and 0 < yc < self.ysize:
            i = iy*self.n+ix
            # self.right_imageFrame.itemconfig(self.cells[i], fill="blue",stipple="gray50")
            coord = self.right_cell_state[i]["coordinates"]
            if not self.right_cell_state[i]["clicked"]:
                self.right_cell_state[i]["clicked"] = True
                im = self.create_translucent_rectangle(self.right_imageFrame,coord[0],
                                                coord[1], coord[2], coord[3], 
                                                fill="red",alpha=0.5)
                self.right_cell_state[i]["rectangle"] = im
            else:
                self.right_cell_state[i]["clicked"] = False
                self.right_imageFrame.delete(self.right_cell_state[i]["rectangle"])
            cpy_r = deepcopy(self.right_cell_state)
            self.global_cell_states[self.current_frame]["right"] = cpy_r

    def load_saved_states(self,filename):
        with open(filename, 'rb') as f:
            self.global_cell_states = pickle.load(f)

    def save_states(self,filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.global_cell_states, f)
    
    def draw_saved_states(self,frame_number):

        left = self.global_cell_states[frame_number].get("left")
        right = self.global_cell_states[frame_number].get("right")

        if left:
            self.left_cell_state = self.global_cell_states[frame_number].get("left")
        if right:
            self.right_cell_state = self.global_cell_states[frame_number].get("right")
        # print(frame_number,left_cell_state,right_cell_state)
        if self.left_cell_state or self.right_cell_state:
            for i in range(self.n*self.n):
                if self.left_cell_state:
                    left_coord = self.left_cell_state[i]["coordinates"]
                    if self.left_cell_state[i]["clicked"]:
                        im = self.create_translucent_rectangle(self.left_imageFrame,left_coord[0],
                                                        left_coord[1], left_coord[2], left_coord[3], 
                                                        fill="red",alpha=0.5)
                        self.left_rect_dict[i] = im

                if self.right_cell_state:
                    right_coord = self.right_cell_state[i]["coordinates"]
                    if self.right_cell_state[i]["clicked"]:
                        im = self.create_translucent_rectangle(self.right_imageFrame,right_coord[0],
                                                            right_coord[1], right_coord[2], right_coord[3], 
                                                            fill="red",alpha=0.5)
                        self.right_rect_dict[i] = im
        else:
            pass


    # def update_next_sides(self,image):
    # #Update canvas when next is clicked 
    #       # convert image for tkinter
    #     # self.left_panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
    #     # self.left_panel.config(image=imgtk)
    #     im = self.wrap_perspective(image)
    #     #show image opencv
    #     # cv2.imshow("next",im)
    #     # cv2.waitKey(0)
    #     # cv2.destroyWindow("next")
    #     cv2image = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
    #     current_image = Image.fromarray(cv2image)  # convert image for PIL
    #     imgtk = ImageTk.PhotoImage(image=current_image)
    #     # self.left_imageFrame.create_image(0,0,image=self.img, anchor="nw")
    #     self.left_imageFrame.config(image=imgtk)
    #     self.draw_grid(self.left_imageFrame)
    #     # self.right_imageFrame.create_image(0,0,image=self.img, anchor="nw")
    #     self.right_imageFrame.config(image=imgtk)
    #     self.draw_grid(self.right_imageFrame)

    # def update_prev_sides(self,image):
    #      #Update canvas when prev is clicked
    #     # cv2image = cv2.cvtColor(self.door_img, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
    #     # current_image = Image.fromarray(cv2image)  # convert image for PIL
    #     # imgtk = ImageTk.PhotoImage(image=current_image)  # convert image for tkinter
    #     # self.right_panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
    #     # self.right_panel.config(image=imgtk)
    #     im = self.wrap_perspective(image)
    #     cv2image = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
    #     current_image = Image.fromarray(cv2image)  # convert image for PIL
    #     imgtk = ImageTk.PhotoImage(image=current_image)
    #     self.left_imageFrame.imgtk = imgtk # anchor imgtk so it does not be deleted by garbage-collector
    #     self.left_imageFrame.config(image=imgtk)
    #     self.draw_grid(self.left_imageFrame)
    #     self.right_panel.imgtk = imgtk # anchor imgtk so it does not be deleted by garbage-collector
    #     self.right_panel.config(image=imgtk)
    #     self.draw_grid(self.right_imageFrame)


    def update_sides(self,image):
        """ Update side canvas when is clicked """

        im = self.wrap_perspective(image)
        cv2image = cv2.cvtColor(im, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
        current_image = Image.fromarray(cv2image)  # convert image for PIL
        imgtk = ImageTk.PhotoImage(image=current_image)

        self.left_imageFrame.imgtk = imgtk
        self.left_imageFrame.create_image(0,0,image=imgtk, anchor="nw")
         # anchor imgtk so it does not be deleted by garbage-collector
        # self.left_imageFrame.config(image=imgtk)
        self.draw_grid_left()


        self.right_imageFrame.imgtk = imgtk # anchor imgtk so it does not be deleted by garbage-collector
        self.right_imageFrame.create_image(0,0,image=imgtk, anchor="nw")
        # self.right_imageFrame.config(image=imgtk)
        self.draw_grid_right()

    def prev_frame(self,*_):
        """ Get previous frame from the video stream """
        if len(self.prev_images) > 0:
            self.current_frame -= 1
            self.current_image = self.prev_images.pop()
            # self.update_prev_sides(self.current_image)
            self.update_sides(self.current_image)
            # self.update_right(self.current_image)
            cv2image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.center_panel.imgtk = imgtk
            self.center_panel.config(image=imgtk)

            self.draw_saved_states(self.current_frame)
            self.left_imageFrame.create_text(20,20,text=str(self.current_frame),anchor="nw")

            # self.left_cell_state = self.global_cell_states[self.current_frame]["left"]
            # self.right_cell_state = self.global_cell_states[self.current_frame]["right"]

    def copy_prev_frame_state(self,*_):
        """ Copy previous frame state to the current frame """
        # self.next_frame()
        # print(self.current_frame)
       
        prev_state = deepcopy(self.global_cell_states[self.current_frame-1])
        # print(prev_state["left"][0])

        self.global_cell_states[self.current_frame] = prev_state
        # self.update_sides(self.current_image)
        self.draw_saved_states(self.current_frame)
        self.global_cell_states[self.current_frame] = deepcopy(self.global_cell_states[self.current_frame-1])

        # self.global
        # self.next_frame()
        # self.left_imageFrame.create_text(20,20,text=str(self.current_frame),anchor="nw")


    def next_frame(self,*_):
        """ Get frame from the video stream and show it in Tkinter """
        ok, frame = self.vs.read()  # read frame from video stream
        if ok:  # frame captured without any errors
            self.current_frame += 1
            self.prev_images.append(frame)
            # self.update_next_sides(frame)
            self.update_sides(frame)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  # convert colors from BGR to RGBA
            self.current_image = Image.fromarray(cv2image)  # convert image for PIL
            imgtk = ImageTk.PhotoImage(image=self.current_image)  # convert image for tkinter
            self.center_panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            self.center_panel.config(image=imgtk)  # show the image


                
            # print("after next",self.current_frame-1,self.global_cell_states[self.current_frame-1].get("left"))
            # self.global_cell_states[self.current_frame]["left"] = self.left_cell_state
            # self.global_cell_states[self.current_frame]["right"] = self.right_cell_state
            #display frame number
            self.left_imageFrame.create_text(20,20,text=str(self.current_frame),anchor="nw")

    
    def destructor(self):
        """ Destroy the window object and release all resources """
        print("[INFO] closing...")
        self.window.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application

# construct the argument parse and parse the arguments


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="./",
    help="path to output directory to store snapshots (default: current folder")
args = vars(ap.parse_args())

# start the app
print("[INFO] starting...")

pba = Application(args["output"])
pba.window.mainloop()