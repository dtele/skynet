import matplotlib.pyplot as plt


class PlotDisplay:
    def __init__(self, bg_img):
        self.fig, self.ax = plt.subplots()
        self.bg_img = bg_img
        self.ax.imshow(self.bg_img)
        self.ax.axis('off')
        self.line_builder = self.LineBuilder(self)

    class LineBuilder:
        def __init__(self, plot_display):
            self.super_fig = plot_display.fig
            self.super_ax = plot_display.ax

            self.coords = ()
            self.first_cid = self.super_fig.canvas.mpl_connect('button_press_event', self.first_click)
            plt.waitforbuttonpress()

            self.line, = self.super_ax.plot([self.coords[0]], [self.coords[1]])
            self.x_points = list(self.line.get_xdata())
            self.y_points = list(self.line.get_ydata())
            self.count = 1

            self.next_cid = self.line.figure.canvas.mpl_connect('button_press_event', self.next_clicks)

        def get_points(self):
            return list(zip(self.x_points[:4], self.y_points[:4]))

        def first_click(self, event):
            self.coords = event.xdata, event.ydata
            self.super_ax.plot(self.coords[0], self.coords[1], "+", color="black", zorder=10)
            plt.disconnect(self.first_cid)

        def next_clicks(self, event):
            self.x_points.append(event.xdata)
            self.y_points.append(event.ydata)

            self.super_ax.plot(self.x_points[-1], self.y_points[-1], "+", color="black", zorder=10)
            self.line.set_data(self.x_points, self.y_points)
            self.line.set_color("red")
            self.line.figure.canvas.draw()
            self.count += 1

            if self.count == 4:
                plt.disconnect(self.next_cid)

                self.x_points.append(self.x_points[0])
                self.y_points.append(self.y_points[0])
                self.line.set_data(self.x_points, self.y_points)
                self.line.set_color("red")
                self.line.figure.canvas.draw()

                plt.close(self.super_fig)
