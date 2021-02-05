import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



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
        ax.axes.set_xlim3d(left=self.minx, right=self.maxx)
        ax.axes.set_ylim3d(bottom=self.miny, top=self.maxy)
        ax.axes.set_zlim3d(bottom=self.minz, top=self.maxz)

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

    def show(self, save_name=""):
        plt.show(block=False)
        if(save_name != ""):
            plt.savefig(save_name)

    def clear(self):
        self.ax.cla()


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