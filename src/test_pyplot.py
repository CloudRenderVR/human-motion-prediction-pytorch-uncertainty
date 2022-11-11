import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class AnActuallySaneWayOfDrawingThings():
    def __init__(self, axis, minx, miny, minz, maxx, maxy, maxz):
        self.minx = minx
        self.miny = miny
        self.minz = minz
        self.maxx = maxx
        self.maxy = maxy
        self.maxz = maxz
        self.ax = axis

    def __set_bounds(self):
        self.ax.axes.set_xlim3d(left=self.minx, right=self.maxx)
        self.ax.axes.set_ylim3d(bottom=self.miny, top=self.maxy)
        self.ax.axes.set_zlim3d(bottom=self.minz, top=self.maxz)

    def draw_points(self, points, colors):
        self.__set_bounds()
        pass

    def draw_lines(self, lines, colors=[]):
        self.__set_bounds()
        i = 0
        for line in lines:
            color_to_draw = (0, 0, 0)
            if i < len(colors):
                color_to_draw = colors[i]
            self.ax.plot(   [line[0][0], line[1][0]],
                            [line[0][1], line[1][1]],
                         zs=[line[0][2], line[1][2]],
                         color = color_to_draw)
            i+=1

    def draw_look_direction(self,lines):
        x_diff = 0
        y_diff = 0
        z_diff = 0
        self.__set_bounds()
        i = 0
        rad = 200
        #degree from head is in slope form
        degree_from_head = 1/.8391
        for line in lines:
            distance = math.sqrt((line[1][0]-line[0][0])**2+(line[1][1]-line[0][1])**2+(line[1][2]-line[0][2])**2)
            color_to_draw = (0, 0, i)
            self.ax.plot(   [line[0][0], line[1][0]],
                            [line[0][1], line[1][1]],
                         zs=[line[0][2], line[1][2]],
                         color = color_to_draw)
            if i == 1:
                color_to_draw = (i, 0, 0)
                #calculate the differences for the head vector
                #distance calc
                #calculate the z_diff height for 50 degree depending on z val
                
                
                #calculate the differences based on slope
                x_diff = line[1][0]-rad*((line[1][0]-line[0][0])/distance)
                y_diff = line[1][1]-rad*((line[1][1]-line[0][1])/distance)
                #sin 50 deg = 0.766044443119
                #distance from original z to new z
                z_diff = line[1][2]-rad*((line[1][2]-line[0][2])/(distance))
                z_dist = math.sqrt((line[1][0]-x_diff)**2+(line[1][1]-y_diff)**2+(line[1][2]-z_diff)**2) 
                z_diff = z_diff + 0.766044443119*z_dist 
                
                
                #x_diff = line[0][0]
                #y_diff = line[0][1]
                z_diff = line[0][2]+0.766044443119*distance
                """
                x_diff = (line[0][0]/abs(line[0][0])) * (0.766044443119*rad+abs(line[0][0]-line[1][0]))
                y_diff = (line[0][1]/abs(line[0][1])) * (0.766044443119*rad+abs(line[0][1]-line[1][1]))
                z_diff = line[0][2]
                """
                #new position will use these differences
                
                self.ax.plot([line[1][0], x_diff],
                             [line[1][1], y_diff],
                             zs=[line[1][2], z_diff],
                             color = color_to_draw)
                
            i+=1
            
        return ([x_diff,y_diff,z_diff])
    def draw_look_direction_zeroed(self,lines):
        color_to_draw = (0, 0, 1)
        line = lines[1]
        x_diff = 0
        y_diff = 0
        z_diff = 0
        self.__set_bounds()
        i = 0
        rad = 200
        #degree from head is in slope form
        degree_from_head = 1/.8391
        #calculate the differences for the head vector
        #distance calc
        distance = math.sqrt((line[1][0]-line[0][0])**2+(line[1][1]-line[0][1])**2+(line[1][2]-line[0][2])**2)
        #calculate the z_diff height for 50 degree depending on z val
                
                
        #calculate the differences based on slope
        x_slope = (line[1][0]-line[0][0])/(line[1][1]-line[0][1])
        angle = math.atan(x_slope)
        x_diff = rad*math.cos(angle)
        y_diff = rad*math.sin(angle)
        #x_diff = rad*((line[1][0]-line[0][0])/distance)
        #y_diff = rad*((line[1][1]-line[0][1])/distance)
        
        
        
        #sin 50 deg = 0.766044443119
        #distance from original z to new z
        #z_diff = 0-rad*((line[1][2]-line[0][2])/(distance))
        #z_dist = math.sqrt((line[1][0]-x_diff)**2+(line[1][1]-y_diff)**2+(line[1][2]-z_diff)**2) 
        #z_diff = z_diff + 0.766044443119*z_dist 
                
        #TODO: edit so it resembles a sphere rather than a cyliner
        
        
        z_diff = math.atan(line[1][2]-line[0][2]/distance) #calculate the degrees of the slope
        z_diff = z_diff #add angle between head and eye direction
        z_diff = rad*math.sin(z_diff)
        
        z_diff = math.atan((0.766044443119*distance+line[0][2]-line[1][2])/rad)
        z_diff = math.sin(z_diff)*rad
        """
        x_diff = (line[0][0]/abs(line[0][0])) * (0.766044443119*rad+abs(line[0][0]-line[1][0]))
        y_diff = (line[0][1]/abs(line[0][1])) * (0.766044443119*rad+abs(line[0][1]-line[1][1]))
        z_diff = line[0][2]
        """
        #new position will use these differences
                
        self.ax.plot([0, x_diff],
                     [0, y_diff],
                     zs=[0, z_diff],
                     color = color_to_draw)
    
        return ([x_diff,y_diff,z_diff])

    def show(self, save_name=""):
        self.ax.set_axis_off()
        plt.show(block=False)
        # TEMP(andrew): don't both saving to pngs
        #if(save_name != ""):
        #    plt.savefig(save_name)

    def clear(self):
        self.ax.cla()

def __main__():
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  test_drawer = AnActuallySaneWayOfDrawingThings(ax, -3, -3, -3, 3, 3, 3)
  test_drawer.draw_lines([[[1,1,1], [2,2,2]],
                          [[1,2,3], [3,2,1]],
                          [[-1,-1,-1], [1,1,1]]],
                         [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
  test_drawer.show()
  plt.pause(2)
  test_drawer.clear()
  
  test_drawer.draw_lines([[[1,2,1], [2,3,2]],
                          [[1,3,3], [3,3,1]],
                          [[-1,0,-1], [1,2,1]]],
                         [(1, 0.5, 0), (0.5, 1, 0), (0.5, 0, 1)])

  test_drawer.show()
  plt.pause(2)
  test_drawer.clear()
