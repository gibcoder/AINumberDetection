from tkinter import Tk,Canvas
from PIL import Image, ImageTk, ImageGrab, ImageOps, ImageDraw
import os
import numpy as np
class DrawingApp:
    def __init__(self, tkInstance):
        self.tkInstance = tkInstance
        self.tkInstance.title("28x28 Drawing Canvas")
        
        self.pixel_size=10
        self.canvas_size=280
        self.brush_size = 10 

        self.last_x, self.last_y = None, None

        self.canvas=Canvas(tkInstance,width=self.canvas_size, height=self.canvas_size, bg="white")
        self.canvas.pack()

        self.canvas.bind("<B1-Motion>",self.motion_detected)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "white")
        self.draw = ImageDraw.Draw(self.image)

    def motion_detected(self,event):
        event_x=event.x
        event_y=event.y

        length=self.brush_size

        x0=event.x-length
        x1=event.x+length
        y0=event.y-length
        y1=event.y+length
        
        print(x0,y0)
        self.canvas.create_oval(x0,y0,x1,y1,fill='black')
        self.draw.ellipse([x0, y0, x1, y1], fill='black')
        self.last_x, self.last_y = event_x, event_y

        


    def reset(self, event):
        self.last_x, self.last_y = None, None
        self.saveImage()
    #if os.path.exists("drawing.png"):
    #    os.remove("drawing.png")
    #if os.path.exists("drawing_28x28.png"):
    #    os.remove("drawing_28x28.png")
    def saveImage(self):
        
        x = self.tkInstance.winfo_rootx() + self.canvas.winfo_x()
        y = self.tkInstance.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        ImageGrab.grab().crop((x, y, x1, y1)).save("drawing.png")
        img = Image.open("drawing.png")
        img = img.convert("L")
        resized_img = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        bw_img = resized_img.point(lambda x: 0 if x < 128 else 255, '1')
        bw_img.save("drawing_28x28.png")
       

        image_array = np.array(bw_img, dtype=np.uint8)
        print("28x28 Pixel Values:")
        print(image_array)

        if os.path.exists("drawing.png"):
            os.remove("drawing.png")



    
        

tkInstance = Tk()
app = DrawingApp(tkInstance)
tkInstance.mainloop()