save_screens = False
save_freq = 5 #Save once in x frames

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
is_segmented = False

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
