# Resolutions

server_height = 360
server_width = 360
render_height = 360
render_width = 360

# 96 * 8/3 = 256
# 3/8 = 0.375

scale_factor = 0.375

'''
Segmentation Classes:
0	None
1	Buildings
2	Fences
3	Other
4	Pedestrians
5	Poles
6	RoadLines
7	Roads
8	Sidewalks
9	Vegetation
10	Vehicles
11	Walls
12	TrafficSigns

Source: https://carla.readthedocs.io/en/latest/cameras_and_sensors/
'''
no_of_classes = 13
class_grouping = [4, 5, 12, 6, 7, 8, 10, (0, 1, 2, 3, 9, 11)]

colors_segment = [[0, 0, 0], #Black None
            [100, 100, 255], #X-colored Buildings
            [246, 0, 255], #Dark pink Fences
            [0, 255, 251], #Cyan Other stuffs
            [255, 0, 0], #Red Pedestrians
            [255, 255, 255], #White Poles
            [255, 142, 255], #Pink Roadlines
            [119, 120, 118], #Gray Roads
            [255, 153, 0], #Orange Sidewalks
            [30, 0, 255], #Dark blue Vegetation
            [0, 255, 34], # Light Green Vehicles
            [0, 149, 255], #Light blue Walls
            [255, 251, 0], #Yellow TrafficSigns
            ]

assert(len(colors_segment)==no_of_classes)
