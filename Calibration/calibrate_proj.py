import math
from multiprocessing import pool
import numpy as np
import cv2
from cv2 import aruco
from itertools import permutations  
import json

projected_coords = [[450,500],[450,800], [1300, 550], [1350,850]]
#projected_coords = [[400,300],[400,780],[1520,300],[1520,780]] #Original test coords
screen_coords=[[0,0], [0,1080],[1920,0],[1920,1080]]
sleeping_time = 300

############### Getting ancient calibration data ###############

# try:
#     with open('core/calibration/calibration_data.json', 'r') as f:
#         data = json.load(f)

#     for k,v in data.items():
#         globals()[k]=np.array(v)
    
# except:
pool_coords = screen_coords
detected_coords = projected_coords
    

############### Window Configuration ###############
cv2.namedWindow("Pool", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Pool", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


############### Camera Configuration ###############

CAM_NUMBER = 1 #default
# try:
#     with open("home/config.json", "r") as f:
#                     config = json.load(f)
#                     if ("camera" in config and "number" in config["camera"]):
#                         CAM_NUMBER = config["camera"]["number"]
# except:
#     print("No config file found, using default camera number")

# with open("home/calibration_data.json", "r") as f:
#     data = json.load(f)

camera_distortion = np.load('./data/dist.npy')

def get_frame():

    cap = cv2.VideoCapture(CAM_NUMBER)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    ok, frame = cap.read()

    cap.release()
    
    if not ok:
        cv2.destroyAllWindows()
        print("No camera detected")
        input()
        exit()
    
    return frame

############### place circle with mouseCallBack event ###############

def draw_circle(event,x,y,flags,param):
    # print('x',x,'y',y)
    global l_circle, background
    
    if event in [cv2.EVENT_RBUTTONDOWN, cv2.EVENT_LBUTTONDOWN]:
        
        if len(l_circle)>=4:
            l_circle[min(enumerate([(xC-x)**2+(yC-y)**2 for xC,yC in l_circle]), key=lambda x: x[1])[0]]=[x,y]
        else:
            l_circle+=[[x,y]]
        # print(l_circle)
        frame=background.copy()
        
        for i,p1 in enumerate(l_circle):
            p1=tuple(p1)
            for p2 in l_circle[i+1:]:
                p2=tuple(p2)
                cv2.line(frame,p1,p2,(255,0,0),3)
            cv2.circle(frame,p1,20,(0,0,255),-1)
        
        cv2.imshow('Pool',frame)


############### Pool Calibration ###############

background=get_frame()
cv2.imshow('Pool',background)
cv2.waitKey(sleeping_time)

# blur image
gaussian_blur = cv2.GaussianBlur(background, (7,7), 0)

# apply HSV mask to highlight green
hsv = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)


hsv_lower = (40, 40,40)
hsv_higher= (70, 255,255)
# hsv_lower = (46,10,10) # lower values for green mask
# hsv_higher = (86,255,255) # max values for green mask

mask = cv2.inRange(hsv, hsv_lower, hsv_higher)
green = cv2.bitwise_and(gaussian_blur, gaussian_blur, mask = mask)
cv2.imshow("Pool", green)
cv2.waitKey(sleeping_time)

# image processing to get a thresed blob
gray = cv2.cvtColor(green, cv2.COLOR_RGB2GRAY)
ret,thresh = cv2.threshold(gray,50,255,0)
kernel = np.ones((5,5), np.uint8)
dilate = cv2.dilate(thresh, kernel, iterations=1)
cv2.imshow("Pool", dilate)
cv2.waitKey(sleeping_time*2)


# ---------------- STEP 2 ----------------

main_line_number = 6
lines = []

class Line:
    def __init__(self, x1 : np.float32, y1 : np.float32, x2 : np.float32, y2 : np.float32) -> None:
        self.x1 = int(x1)
        self.y1 = int(y1)
        self.x2 = int(x2)
        self.y2 = int(y2)
        self.length = math.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

        self.a = (y2 - y1) / (x2 - x1)
        self.b = y1 - self.a * x1
    
    def __str__(self) -> str:
        str = (f'Coeff : {self.a} and Line : A({self.x1}, {self.y1}) to B({self.x2}, {self.y2})')
        return str

fld = cv2.ximg
fld = cv2.ximgproc.createFastLineDetector().detect(dilate)
if fld is not None:
    for line in fld:
        new_line = Line(line[0][0],line[0][1], line[0][2], line[0][3])
        lines.append(new_line)

lines.sort(key = lambda x: x.length, reverse=True) #sort lines by length

# Draw the line for visualisation
for line in lines[:main_line_number]:
    cv2.line(background, (line.x1, line.y1), (line.x2, line.y2), (255, 0, 255), 5)

cv2.imshow("Pool", background)
cv2.waitKey(sleeping_time)


# ---------------- STEP 3 ----------------

intersection_points = []
index_to_remove = []

def line_intersection(line1, line2): # source : https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


for i in range(len(lines[:main_line_number])):
    for j in range(i+1,len(lines[:main_line_number])):
        if (abs(lines[i].a - lines[j].a) > 0.5): # avoid the calculation of almost parallel lines
            intersection_points.append(line_intersection(((lines[i].x1,lines[i].y1), (lines[i].x2,lines[i].y2)),
            ((lines[j].x1, lines[j].y1), (lines[j].x2,lines[j].y2))))

# keep intersection points on the image size
intersection_points = [[int(i[0]), int(i[1])] for i in intersection_points if (i[0] >= 0 and i[1] >= 0 and i[0] <= background.shape[1] and i[1] <= background.shape[0])]

# remove too close intersection points
for i in range(len(intersection_points)):
    for j in range(i+1,len(intersection_points)):
        if(math.sqrt((intersection_points[j][0] - intersection_points[i][0])**2 + (intersection_points[j][1] - intersection_points[i][1])**2) < 100):
            index_to_remove.append(i)

intersection_points = [intersection_points[a] for a in range(len(intersection_points)) if a not in index_to_remove]

# Draw the circles for visualisation
for i in intersection_points:
    cv2.circle(background, (i[0],i[1]), 6, (2,0,255),-1)

cv2.imshow("Pool", background)
cv2.waitKey(sleeping_time)
print("Pool coords =", intersection_points)
pool_coords = intersection_points

cv2.putText(background,
            "Click on pool 4 corners",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1/2,
            (0,0,255),
            1,
            cv2.LINE_AA)

l_circle=pool_coords
cv2.imshow('Pool', background)

cv2.setMouseCallback('Pool', draw_circle)

frame=background.copy()
for i,p1 in enumerate(l_circle):
    p1=tuple(p1)
    for p2 in l_circle[i+1:]:
        p2=tuple(p2)
        cv2.line(frame,p1,p2,(255,0,0),3)
    cv2.circle(frame,p1,20,(0,0,255),-1)

cv2.imshow('Pool',frame)

cv2.waitKey(0)
while len(l_circle)<4:
    cv2.waitKey(0)
    
cv2.setMouseCallback('Pool', lambda *args: None)
    
pool_coords=l_circle.copy()

frame_camera=background.copy()

############### Calibration Projecteur ###############

def drawArucoFrame():
    #place aruco patters on images at projected_coords coordinates
    arucoFrame=np.full((1080,1920,3), 255,np.uint8)
    aruco0 = cv2.imread("core/calibration/aruco0.png")
    aruco1 = cv2.imread("core/calibration/aruco1.png")
    aruco2 = cv2.imread("core/calibration/aruco2.png")
    aruco3 = cv2.imread("core/calibration/aruco3.png")
    # aruco4 = cv2.imread("core/calibration/aruco4.png")
    # aruco5 = cv2.imread("core/calibration/aruco5.png")

    arucoFrame[projected_coords[0][1]:projected_coords[0][1]+aruco0.shape[0], projected_coords[0][0]:projected_coords[0][0]+aruco0.shape[1]] = aruco0
    arucoFrame[projected_coords[1][1]:projected_coords[1][1]+aruco1.shape[0], projected_coords[1][0]:projected_coords[1][0]+aruco1.shape[1]] = aruco1
    arucoFrame[projected_coords[2][1]:projected_coords[2][1]+aruco2.shape[0], projected_coords[2][0]:projected_coords[2][0]+aruco2.shape[1]] = aruco2
    arucoFrame[projected_coords[3][1]:projected_coords[3][1]+aruco3.shape[0], projected_coords[3][0]:projected_coords[3][0]+aruco3.shape[1]] = aruco3
    # arucoFrame[projected_coords[4][1]:projected_coords[4][1]+aruco4.shape[0], projected_coords[4][0]:projected_coords[4][0]+aruco4.shape[1]] = aruco4
    # arucoFrame[projected_coords[5][1]:projected_coords[5][1]+aruco5.shape[0], projected_coords[5][0]:projected_coords[5][0]+aruco5.shape[1]] = aruco5

    return arucoFrame 

def findArucoMarkers(img, markerSize=4, totalMarkers=250,draw=True):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key = getattr(aruco,f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, _ = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    # print(ids,bboxs)
    coords = []
    if (ids is None):
        print("0 aruco patterns detected out of 4")
    elif(ids.size != 4):
        print("%s aruco patterns detected out of 4" % ids.size)
    else:
        print("%s aruco patterns detected out of 4" % ids.size)
        #converting NumPy arrays into a int list + sort aruco patterns in order
        ids = [i[0] for i in ids.tolist()]
        coords = [bboxs[i][0][0].tolist() for i in range(len(ids))]
        coords = [[int(a),int(b)] for a,b in coords]
        sorted_pairs = sorted(zip(ids, coords))
        tuples = zip(*sorted_pairs)
        ids, coords = [ list(tuple) for tuple in  tuples]
        # print(ids)
    # print("coords : ", coords)
    return coords

background = drawArucoFrame()
cv2.imshow('Pool',background)
cv2.waitKey(sleeping_time)

frame = get_frame()
coords = findArucoMarkers(frame)

if len(coords) == 4:
    detected_coords = coords[:4]
else: #select manually
    print()
    print('Manual Calibration required : click on top-right of Aruco square')
    for x,y in pool_coords:
        cv2.circle(frame, (x,y), 20, (0,0,255), -1)

    l_circle=detected_coords
    cv2.imshow('Pool',frame)

    cv2.setMouseCallback('Pool',draw_circle)

    background = frame.copy()
    for i,p1 in enumerate(l_circle):
        p1=tuple(p1)
        for p2 in l_circle[i+1:]:
            p2=tuple(p2)
            cv2.line(frame,p1,p2,(255,0,0),3)
        cv2.circle(frame,p1,20,(0,0,255),-1)

    cv2.imshow('Pool',frame)

    cv2.waitKey(0)
    while len(l_circle)<4:
        cv2.waitKey(0)

    cv2.setMouseCallback('Pool', lambda *args: None)
        
    detected_coords=l_circle.copy()

    frame_projector = background.copy()
    for p1 in detected_coords:
        cv2.circle(frame_projector, tuple(p1), 40, (0, 0, 255), -1)
    
############### Ordering points and format ###############

def ordering(l_point1, l_point2):

    global background
    
    l_distance_ordered=[]
    for l_ordering in permutations(list(range(len(l_point1)))):
        d=0
        l_ordered_point=[]
        for i1,i2 in enumerate(l_ordering):
            d+=np.linalg.norm(np.array(l_point1[i1])
                              - np.array(l_point2[i2]))
            l_ordered_point+=[l_point2[i2]]
        
        l_distance_ordered+=[[d,l_ordered_point]]

    l_ordered_point=min(l_distance_ordered, key=lambda x: x[0])[1]

    frame=background.copy()
    
    for p1,p2 in zip(l_point1, l_ordered_point):
        p1=tuple(p1)
        p2=tuple(p2)
        cv2.line(frame,p1,p2,(255,0,0),3)
        cv2.circle(frame,p1,20,(0,0,255),-1)
        cv2.circle(frame,p2,20,(0,255,0),-1)

    cv2.imshow('Pool',frame)
    cv2.waitKey(1000)
    
    return l_ordered_point

pool_coords = ordering(detected_coords, pool_coords)
screen_coords = ordering(pool_coords, screen_coords)
projected_coords =  ordering(pool_coords, projected_coords)

screen_coords = [[i[0],i[1]] for i in screen_coords]
pool_coords = [[i[0],i[1]] for i in pool_coords]
projected_coords = [[i[0],i[1]] for i in projected_coords]
detected_coords = [[i[0],i[1]] for i in detected_coords]
# screen_coords = [[i[0]-960,i[1]-540] for i in screen_coords]
# pool_coords = [[i[0]-960,i[1]-540] for i in pool_coords]
# projected_coords = [[i[0]-960,i[1]-540] for i in projected_coords]
# detected_coords = [[i[0]-960,i[1]-540] for i in detected_coords]

detected_coords=np.float32(detected_coords)
pool_coords=np.float32(pool_coords)
screen_coords=np.float32(screen_coords)
projected_coords=np.float32(projected_coords)

# print("---- Debug ----")
# print("screen coords : ", screen_coords)
# print("pool_coords : ", pool_coords)
# print("detected_coords : ", detected_coords)
# print("projected_coords : ",projected_coords)

############### Set projection matrix ###############
#                                           IN              OUT
tMat1_0 = cv2.getPerspectiveTransform(screen_coords, projected_coords)
tMat1_1 = cv2.getPerspectiveTransform(detected_coords, projected_coords)
tMat1_2 = cv2.getPerspectiveTransform(projected_coords, pool_coords)
projection_matrix = tMat1_1.dot(tMat1_2).dot(tMat1_0)

poolFocus_matrix = cv2.getPerspectiveTransform(pool_coords, screen_coords)

############### Calibration Test ###############

calibration_test2=np.zeros((1080,1920,3), np.uint8)
for i,p1 in enumerate(screen_coords):
    p1=tuple(map(int, p1))
    for p2 in screen_coords[i+1:]:
        p2=tuple(map(int, p2))
        cv2.line(calibration_test2,p1,p2,(255,255,255),3)
    cv2.circle(calibration_test2,p1,40,(255,255,255),-1)

cv2.putText(calibration_test2,
            "Image 2D a deformer dans un espace 3D pour fitter le Pool, auppuyez sur une touche pour continuer",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1/2,
            (0,0,255),
            1,
            cv2.LINE_AA)
cv2.imshow('Pool', calibration_test2)
# cv2.waitKey(0)

test_1 = cv2.warpPerspective(calibration_test2, projection_matrix, (1920,1080), flags=cv2.INTER_LINEAR)
cv2.putText(test_1,
            "Image 2D deforme dans un espace 3D, auppuyez sur une touche pour continuer",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1/2,
            (0,0,255),
            1,
            cv2.LINE_AA)
cv2.imshow('Pool',test_1)
# cv2.waitKey(0)

test_2 = cv2.warpPerspective(frame_camera, poolFocus_matrix, (1920,1080), flags=cv2.INTER_LINEAR)
cv2.putText(test_2,
            "Image 3D du Pool déformer pour passer dans l'espace 2D de l'écran, auppuyez sur une touche pour continuer",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1/2,
            (0,0,255),
            1,
            cv2.LINE_AA)
cv2.imshow('Pool',test_2)

# camera_distortion = np.float32([[ 9.84936576e-01, -3.66695373e-03, -2.30536946e+00],
#        [ 4.03649320e-03,  9.86309813e-01,  1.12314735e+01],
#        [-1.58822878e-06,  7.31531798e-07,  1.00000000e+00]])

outpts = []
for x,y in screen_coords:
    x = (projection_matrix[0][0] * x + projection_matrix[0][1] * y + projection_matrix[0][2]) / (projection_matrix[2][0] * x + projection_matrix[2][1] * y + projection_matrix[2][2])
    y = (projection_matrix[1][0] * x + projection_matrix[1][1] * y + projection_matrix[1][2]) / (projection_matrix[2][0] * x + projection_matrix[2][1] * y + projection_matrix[2][2])
    x -= 960
    y -= 540
    x = (camera_distortion[0][0] * x + camera_distortion[0][1] * y + camera_distortion[0][2]) / (camera_distortion[2][0] * x + camera_distortion[2][1] * y + camera_distortion[2][2])
    y = (camera_distortion[1][0] * x + camera_distortion[1][1] * y + camera_distortion[1][2]) / (camera_distortion[2][0] * x + camera_distortion[2][1] * y + camera_distortion[2][2])
    outpts.append([int(x),int(y)])
outpts = np.float32(outpts)    

############### Export Data in json file ###############
d_information={"projection_matrix": projection_matrix,
               "poolFocus_matrix": poolFocus_matrix,
               "detected_coords": detected_coords.astype('int'),
               "pool_coords": pool_coords.astype('int'),
               "screen_coords": screen_coords.astype('int'),
               "projected_coords": projected_coords.astype('int'),
               "camera_distortion": camera_distortion,
               "outpts": outpts
        }

d_information={k:v.tolist() for k,v in d_information.items()}

with open('home/calibration_data.json', 'w') as f:
    json.dump(d_information, f, indent=4)

print("Calibration terminée avec succès!")

cv2.destroyAllWindows()
exit()