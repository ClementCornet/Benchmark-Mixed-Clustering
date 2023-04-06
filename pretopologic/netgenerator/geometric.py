from scipy.spatial import distance_matrix
import numpy as np
import math
import pandas as pd

def qualitative(x, y):
    return pd.Series([int(x != y)])

def diff(x, y):
    return abs(x - y)

# We need to be using dm only so we are consistent with the distance calculation method
def areaminmax(dm, df_data):
    # x_dif = df_data[0].max() - df_data[0].min()
    # y_dif = df_data[1].max() - df_data[1].min()
    # # print('df_data[0].max(): ', df_data[0].max())
    # return x_dif * y_dif
    area = 1
    for i in range(df_data.shape[1]):
        print(f"df_data.iloc[:, {i}].max() = {df_data.iloc[:, i].max()}")
        print(f"df_data.iloc[:, {i}].min() = {df_data.iloc[:, i].min()}")
        area *= (df_data.iloc[:, i].max() - df_data.iloc[:, i].min())
    return area


def area(dm, data):
    return (dm.max()**2)


# # returns the adjacency matrix of the elements that are closer than the "average distance",
# # normalized by the biggest distance
# # c.f. calculate average distance
# # points is a list of tuples with the two coordinates
# def network_ball(dm, radius):
#     adj = (dm < radius).astype('float')
#     # With this parametrisation works fine for the 6 clusters example, besides a lonely small cluster
#     # adj = (dm < 1.12*th).astype('float')
#     np.fill_diagonal(adj, 0)
#     return adj


def network_ball_distances(dm, radius):
    adj = (dm > radius)
    dm[adj] = 0
    network = dm/radius
    network = 1 - network
    network[adj] = 0
    return network


# #USED TO NOT HAVE TO MAKE CALCULATION WITH ALL THE NODES ?
# # calculates an approximation of the distance between elements if they were evenly spaced
# # uses the augmenting lines in the grid heuristic
# def prenetwork_closest(points, distances=True):
#     x_dif = np.amax(points[:, 0]) - np.amin(points[:, 0])
#     y_dif = np.amax(points[:, 1]) - np.amin(points[:, 1])
#     area = x_dif*y_dif
#     square_length = math.sqrt(area / len(points))
#
#     dm = distance_matrix(points, points)
#     print('dm: ', dm)
#     closest = np.sum(np.ma.masked_less(dm, square_length/2).mask.astype('float'), axis=1)
#     inverse = (closest-1)/closest
#     real_points = len(points) - sum(inverse)
#
#     # Paramaters: th: 2.5 and degree 4 gives the perfect clusters for HBSDCAN parameters
#     # with radius = 2*square and th = len(points)/real_points works perfect for the 6 clusters
#     # radius = 2*square_length
#     # th = len(points)/(1.2*real_points)
#
#     # with radius = 2*square and th = len(points)/(3*real_points) works very well for the others
#     # radius = 2*square_length
#     # th = len(points)/(3*real_points)
#
#     # with radius = 2.5*square and th = len(points)/(2*real_points) also works very well for the others
#     radius = 2.5*square_length
#     th = len(points)/(2*real_points)
#
#
#     print("")
#     print("")
#     print("AREA: " + str(area))
#     print("Number of points: " + str(len(points)))
#     print("Square length: " + str(square_length))
#     print("REAL_POINTS: " + str(real_points))
#     print("RADIUS: " + str(radius))
#     print("TH: " + str(th))
#
#     if distances:
#         network = network_ball_distances(dm, radius)
#     else:
#         network = network_ball(dm, radius)
#     return Prenetwork(network, [th])
#     # return Prenetwork(network, [2.5])