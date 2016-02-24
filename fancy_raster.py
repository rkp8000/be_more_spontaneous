from __future__ import division, print_function
import numpy as np


def by_row_and_color(ax, spike_trains, drives, labels):
    """
    Plot raster plot where nodes are labeled by row and color.
    Spikes are given by vertical bars, drives by horizontal bars.

    :param spike_trains: multi-cell spike train, rows are timepoints, cols are cells
    :param drives: multi-cell drive, rows are timepoints, cols are cells
    :param labels: list of tuples, one per cell, containing (row, color) of cell.
    """

    for spikes, drive, (row, color) in zip(spike_trains.T, drives.T, labels):

        # convert spikes to plottable x, y pair
        spike_times = spikes.nonzero()[0]
        nans = np.nan * np.zeros(spike_times.shape, dtype=float)

        x = np.array([spike_times, spike_times, nans]).T.flatten()

        y_bottom = -0.3 * np.ones(spike_times.shape, dtype=float) - row
        y_top = 0.3 * np.ones(spike_times.shape, dtype=float) - row

        y = np.array([y_bottom, y_top, nans]).T.flatten()

        ax.plot(x, y, color=color, lw=2)

        # convert drives to plottable x, y pair
        drive_times = drive.nonzero()[0]
        nans = np.nan * np.zeros(drive_times.shape, dtype=float)

        x_left = drive_times - 0.3
        x_right = drive_times + 0.3

        x = np.array([x_left, x_right, nans]).T.flatten()
        y = -row * np.ones(x.shape)

        ax.plot(x, y, color=color, lw=2)


def by_row(ax, spikes, drives):
    """
    Plot raster plots with spikes given by black vertical bars, drives by red horizontal bars.
    Line thicknesses indicate spike/drive amplitudes.
    :param ax: axis object
    :param spikes: multi-cell spike train, rows are timepoints, cols are cells
    :param drives: multi-cell drive, rows are timepoints, cols are cells
    """

    spike_times, spike_rows = spikes.nonzero()

    for time, row in zip(spike_times, spike_rows):

        # define x and y coordinates for vertical line
        xs = [time, time]
        ys = [row - 0.3, row + 0.3]

        ax.plot(xs, ys, c='k', lw=spikes[time, row], zorder=1)

    if drives is not None:
        drive_times, drive_rows = drives.nonzero()

        for time, row in zip(drive_times, drive_rows):

            # define x and y coordinates for horizontal line
            xs = [time - 0.3, time + 0.3]
            ys = [row, row]

            ax.plot(xs, ys, c='r', lw=drives[time, row], zorder=0)