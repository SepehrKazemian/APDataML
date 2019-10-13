import numpy as np
import Plot as plot
import math
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as offline

def plottingDataChunks(data):
    intervalMinutes = int(input("how long is your intervals in minute? "))
    timeIndexValue = int(input("which timeInterval in a day you want to plot? "))
    plotDataFrame = data.loc[data["timeIndex"] == timeIndexValue]
    data_CU_to_numpy = plotDataFrame["CU"].to_numpy()
    data_day_to_list = plotDataFrame["weekDay"].to_list()

    day_changes_index = []
    initialDay = data_day_to_list[0]
    for i in range(len(data_day_to_list)):
        if initialDay != data_day_to_list[i]:
            day_changes_index.append(i)
            initialDay = data_day_to_list[i]

    dividerBoundary = [-1,0,0,0,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,0,0,0,-2]
    for j in range(len(day_changes_index) - 1, 0, -1):
        data_CU_to_numpy = np.insert(data_CU_to_numpy, day_changes_index[j], dividerBoundary)

    scatterArray = []
    fig = go.Figure()
    prevPos = 0
    for i in range(data_CU_to_numpy.shape[0]):
        if data_CU_to_numpy[i] == -1:
            timePlot = [x for x in range(prevPos, i - 1)]
            scatterArray.append(go.Scatter(x = timePlot, y = data_CU_to_numpy[prevPos: i-1], marker_color='blue'))

        if data_CU_to_numpy[i] == -2:
            timePlot = [x for x in range(i - len(dividerBoundary) + 1, i + 1)]
            scatterArray.append(go.Scatter(x = timePlot, y = data_CU_to_numpy[i - len(dividerBoundary) + 1: i + 1], marker_color='red'))
            prevPos = i + 1

    fileName = str(intervalMinutes) + "minInterval" + str(timeIndexValue) + "th.html"
    offline.plot(scatterArray, filename=fileName)
