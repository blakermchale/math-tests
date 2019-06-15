#!/bin/python3

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class GraphViewer:
    def plotKalman(self, positions):
        fig = plt.figure()
        ax = plt.gca(figure=fig)
        ax.set_ylim(0, 15)
        ax.set_xlim(0, 15)
        ax.set_xticks([x for x in range(0, 15)], minor=True)
        ax.plot(positions)
        plt.show()


class VisualSim:
    def __init__(self, world, view_size=None):
        sns.set_style("dark")
        plt.ion()
        # fig = plt.figure()
        self.world = world

        # Fit entire world view if no view size is given
        if view_size is None:
            view_size = self.world.world_size
        elif view_size.size != self.world.dimensions:
            print("View size does not match world dimensions!")
            return
        # Check for when the view size has infinite values and restrict it
        # infvals = np.where(np.isinf(view_size) == True)
        for i in range(view_size.size):
            if np.isinf(view_size[i]):
                view_size[i] = 50.0
                print(view_size[i])
        self.halfview_size = view_size / 2

        self.fig_vissim = plt.figure(0)
        # self.fig_kalman = plt.figure(1)
        self.ax = plt.gca(figure=self.fig_vissim)
        # self.ax = fig.add_subplot(111, projection='3d')
        self.upbounds = self.world.upbounds.points
        self.lowbounds = self.world.lowbounds.points
        self.update()
        # self.ax.set_xticks([x for x in range(int(lowbounds[0]), int(upbounds[0]) + 1)], minor=True)
        # self.ax.set_yticks([y for y in range(int(lowbounds[1]), int(upbounds[1]) + 1)], minor=True)
        # plt.xlim(int(lowbounds[0]), int(upbounds[0]))
        # plt.ylim(int(lowbounds[1]), int(upbounds[1]))

        # # Draw landmarks if they exists
        # if(landmarks is not None):
        #     # loop through all path indices and draw a dot (unless it's at the car's location)
        #     for pos in landmarks:
        #         if(pos != position):
        #             ax.text(pos[0], pos[1], 'x', ha='center', va='center', color='purple', fontsize=20)

        # Display final result

    def update(self):
        # Get position of point to follow
        position = self.world.actors["robot1"].position
        plt.cla()
        # lowbounds_view = self.lowbounds
        # upbounds_view = self.upbounds
        lowbounds_view = position.points - self.halfview_size
        upbounds_view = position.points + self.halfview_size

        lowbound_faults = np.where(
            np.less(
                self.lowbounds,
                lowbounds_view) == False)
        upbound_faults = np.where(
            np.greater(
                self.upbounds,
                upbounds_view) == False)

        for i in lowbound_faults:
            lowbounds_view[i] = self.lowbounds[i]
            upbounds_view[i] = self.lowbounds[i] + 2 * self.halfview_size[i]
        for i in upbound_faults:
            upbounds_view[i] = self.upbounds[i]
            lowbounds_view[i] = self.upbounds[i] - 2 * self.halfview_size[i]
        # # Check if bounds of view go outside world
        # if not np.less(self.lowbounds, lowbounds_view):

        self.ax.set_xticks([x for x in range(
            int(lowbounds_view[0]), int(upbounds_view[0]) + 1)], minor=True)
        self.ax.set_xlim(lowbounds_view[0], upbounds_view[0])
        if self.world.dimensions == 1:
            # self.ax.set_yticks([y for y in range(int(lowbounds_view[1]), int(upbounds_view[1]) + 1)], minor=True)
            self.ax.set_ylim(-1.0, 1.0)
        elif self.world.dimensions == 2:
            self.ax.set_yticks([y for y in range(
                int(lowbounds_view[1]), int(upbounds_view[1]) + 1)], minor=True)
            self.ax.set_ylim(lowbounds_view[1], upbounds_view[1])
        # plt.ylim(int(lowbounds_view[1]), int(upbounds_view[1]))

        self.ax.grid(which='minor', ls='-', lw=1, color='white')

        self.ax.grid(which='major', ls='-', lw=2, color='white')

        if self.world.dimensions == 1:
            self.ax.text(
                position.points[0],
                0,
                'o',
                ha='center',
                va='center',
                color='r',
                fontsize=30)
        elif self.world.dimensions == 2:
            self.ax.text(
                position.points[0],
                position.points[1],
                'o',
                ha='center',
                va='center',
                color='r',
                fontsize=30)
            # if not self.world.count % 10000 == 0:  # When reading count while multithreaded, runs very slow
            #     print(self.world.count)
            #     print("Position: {0}".format(position.points))
        plt.pause(0.001)