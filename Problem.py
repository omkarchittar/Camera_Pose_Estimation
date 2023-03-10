import numpy as np
import cv2
import matplotlib.pyplot as plt

def houghTransform(image, threshold):
    # Image Preprocessing
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
    image_gray = cv2.GaussianBlur(image_gray, (5, 5), cv2.BORDER_DEFAULT) # Apply Gaussian blur to the image

    # Image thresholding based on the pixel intensity
    gray = (image_gray > 205) * image_gray

    # Edge Detection
    blur = cv2.GaussianBlur(gray, (11, 11), cv2.BORDER_DEFAULT)
    kernel = np.ones((5, 5), np.uint8) 
    closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel) # Morphological closing to fill gaps in edges
    edges = cv2.Canny(closing, 180, 200, L2gradient=True) # Detect edges using Canny edge detection

    # Define Hough transform parameters
    theta_values = np.deg2rad(np.arange(0, 180)) # Define the range of theta values to search
    max_d = int(np.ceil(np.sqrt(image.shape[0] ** 2 + image.shape[1] ** 2))) # Calculate the maximum possible distance between a point and the origin
    d_values = np.arange(-max_d, max_d + 1) # Define the range of d values to search

    # Initialize accumulator array with zeroes and appropriate dimensions
    H = np.zeros((len(d_values), len(theta_values)))

    # Iterate over edge pixels and accumulate votes in the accumulator array
    edges_t = np.transpose(np.nonzero(edges))
    for y, x in (edges_t):
        for theta_index, theta in enumerate(theta_values):
            # Calculate the perpendicular distance between the origin and the line passing through the edge point (x,y) with angle theta
            d = x * np.cos(theta) + y * np.sin(theta)
            # Find the index of the d value that is closest to d, and use it to cast a vote in the accumulator array
            d_index = np.argmin(np.abs(d_values - d))
            H[d_index, theta_index] += 1  # Voting

    # Find the peaks in the accumulator array
    peaks = np.argwhere(H > threshold)

    # Find the intersections of the lines corresponding to the peaks
    intersections = []
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            d1, theta1 = peaks[i]
            d2, theta2 = peaks[j]
            # Calculate the slope and y-intercept of the line corresponding to the (d1,theta1) peak
            m1, b1 = -np.cos(theta_values[theta1]) / np.sin(theta_values[theta1]), d_values[d1] / np.sin(theta_values[theta1])
            # Calculate the slope and y-intercept of the line corresponding to the (d2,theta2) peak
            m2, b2 = -np.cos(theta_values[theta2]) / np.sin(theta_values[theta2]), d_values[d2] / np.sin(theta_values[theta2])
            if abs(m1 - m2) < 1e-6:
                continue # Skip if the lines are parallel
            # Calculate the x-coordinate and y-coordinate of the intersection point
            x0 = (b2 - b1) / (m1 - m2)
            y0 = m1 * x0 + b1
            intersections.append((int(x0), int(y0)))

    t = []
    temp = intersections
    for i in range(len(temp) - 1):
        if temp[i][0] > 0 and temp[i][1] > 0:
            t.append((temp[i][0], temp[i][1]))
    p = sorted(t, key=lambda x: x[0])
    q = sorted(t, key=lambda x: x[1])
    corners = [p[0], p[-1], q[0], q[-1]]
    print("Corners : ", corners)
    print("\n")
    return corners

def homography(src_pts, dst_pts):
    assert src_pts.shape == dst_pts.shape, "Input points must have the same shape"
    assert src_pts.shape[0] >= 4, "At least 4 points are required to compute homography"

    num_pts = src_pts.shape[0]

    # Construct the A matrix
    A = np.zeros((2*num_pts, 9))
    for i in range(num_pts):
        x, y = src_pts[i, :]
        u, v = dst_pts[i, :]
        A[2*i, :] = np.array([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A[2*i+1, :] = np.array([0, 0, 0, x, y, 1, -v*x, -v*y, -v])

    # Implement RANSAC to find the best homography
    max_inliers = 0
    best_H = None
    n = 1000
    threshold = 4
    for i in range(n):

        # Randomly select 4 correspondences
        idx = np.random.choice(num_pts, 4, replace=False)
        src_pts_4 = src_pts[idx, :]
        dst_pts_4 = dst_pts[idx, :]

        # Calculate the homography matrix using the 4 correspondences
        A = np.zeros((8, 9))
        for j in range(4):
            x, y = src_pts_4[j, :]
            u, v = dst_pts_4[j, :]
            A[2*j, :] = np.array([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
            A[2*j+1, :] = np.array([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
        _, _, V = np.linalg.svd(A)
        H = V[-1, :].reshape((3, 3))

        # Calculate the number of inliers
        inliers = 0
        for j in range(num_pts):
            src_pt = np.append(src_pts[j], 1)
            dst_pt = np.append(dst_pts[j], 1)
            dst_pt_pred = np.dot(H, src_pt)
            dst_pt_pred /= dst_pt_pred[2]
            error = np.sqrt(np.sum((dst_pt - dst_pt_pred)**2))
            if error < threshold:
                inliers += 1

        # Checking the number of inliers to select the best Homography
        if inliers > max_inliers:
            max_inliers = inliers
            best_H = H

    return best_H

def rotations(R):
    pitch = -np.arcsin(R[2][0])
    roll = np.arctan2(R[2][1]/np.cos(pitch), R[2][2]/np.cos(pitch))
    yaw = np.arctan2(R[1][0]/np.cos(pitch), R[0][0]/np.cos(pitch))
    return roll, pitch, yaw

# --------------------------------------------------
# Comment this section while uncommenting the latter
# --------------------------------------------------
# Read the video file and extract indivisual frames
corners = []
frames = []
vid = cv2.VideoCapture('project2.avi')
success = True
while success:
    success, frame = vid.read()
    if success == True:
        frames.append(frame) 

# Perform Hough Transform on all the Extracted frames to detect the corners
count = 0
for frame in frames:
    count += 1
    print("Reading Frame : ", count)
    corners = houghTransform(frame,100)
    corners.append(corners)
# --------------------------------------------------

# For plotting graphs directly uncomment the corners
# Have stored the co-ordinates from all the frames obtained by once running the code
# corners = [[(802, 465), (1204, 418), (937, 268), (1072, 631)],
# [(801, 464), (1204, 417), (936, 267), (1072, 629)],
# [(803, 463), (1206, 417), (938, 266), (1074, 631)],
# [(802, 464), (1207, 417), (939, 267), (1075, 631)],
# [(802, 466), (1207, 418), (939, 267), (1074, 632)],
# [(803, 466), (1209, 419), (940, 268), (1077, 634)],
# [(804, 468), (1210, 420), (941, 268), (1077, 635)],
# [(805, 467), (1212, 420), (942, 269), (1078, 634)],
# [(805, 467), (1212, 420), (943, 268), (1078, 634)],
# [(806, 465), (1214, 419), (944, 266), (1080, 634)],
# [(806, 462), (1214, 415), (944, 263), (1080, 630)],
# [(805, 460), (1214, 413), (943, 260), (1079, 628)],
# [(806, 458), (1216, 411), (944, 259), (1081, 627)],
# [(805, 457), (1215, 409), (943, 257), (1080, 625)],
# [(807, 454), (1218, 407), (944, 254), (1083, 623)],
# [(808, 451), (1220, 404), (946, 250), (1085, 620)],
# [(805, 447), (1217, 400), (944, 247), (1081, 618)],
# [(805, 446), (1217, 398), (943, 244), (1083, 613)],
# [(805, 443), (1218, 395), (943, 241), (1082, 612)],
# [(805, 441), (1218, 393), (943, 239), (1082, 613)],
# [(806, 441), (1219, 391), (942, 236), (1084, 608)],
# [(805, 439), (1220, 390), (943, 235), (1084, 610)],
# [(806, 438), (1221, 388), (944, 234), (1085, 610)],
# [(805, 434), (1223, 386), (945, 231), (1086, 607)],
# [(805, 433), (1221, 383), (943, 228), (1085, 605)],
# [(805, 432), (1222, 382), (944, 226), (1085, 604)],
# [(804, 432), (1222, 381), (944, 225), (1084, 603)],
# [(802, 431), (1222, 382), (942, 225), (1084, 605)],
# [(804, 433), (1222, 382), (943, 226), (1084, 605)],
# [(804, 431), (1223, 382), (943, 224), (1085, 605)],
# [(804, 431), (1224, 381), (941, 223), (1087, 606)],
# [(806, 434), (1227, 384), (944, 225), (1089, 609)],
# [(806, 435), (1228, 384), (944, 226), (1089, 609)],
# [(807, 436), (1230, 385), (945, 227), (1092, 610)],
# [(808, 435), (1232, 386), (949, 228), (1093, 611)],
# [(807, 436), (1232, 386), (949, 228), (1093, 611)],
# [(807, 435), (1232, 386), (949, 227), (1093, 611)],
# [(804, 433), (1232, 383), (947, 225), (1091, 607)],
# [(802, 431), (1231, 382), (946, 222), (1090, 609)],
# [(804, 431), (1230, 381), (945, 221), (1090, 608)],
# [(802, 430), (1232, 377), (945, 220), (1089, 606)],
# [(801, 429), (1231, 378), (944, 218), (1090, 606)],
# [(802, 427), (1232, 374), (944, 216), (1093, 601)],
# [(801, 425), (1231, 374), (942, 212), (1090, 603)],
# [(801, 424), (1233, 373), (944, 212), (1094, 600)],
# [(802, 423), (1235, 372), (945, 211), (1094, 599)],
# [(799, 422), (1234, 370), (943, 209), (1091, 602)],
# [(800, 421), (1234, 369), (945, 208), (1095, 598)],
# [(799, 418), (1236, 365), (944, 204), (1093, 595)],
# [(798, 415), (1234, 364), (942, 201), (1091, 595)],
# [(797, 412), (1235, 361), (941, 197), (1090, 592)],
# [(795, 408), (1233, 358), (941, 194), (1088, 589)],
# [(794, 407), (1233, 355), (940, 191), (1089, 588)],
# [(796, 405), (1237, 354), (942, 189), (1092, 588)],
# [(797, 404), (1239, 353), (944, 188), (1093, 586)],
# [(800, 403), (1242, 349), (944, 185), (1096, 586)],
# [(802, 402), (1244, 350), (946, 183), (1099, 581)],
# [(801, 399), (1245, 348), (948, 182), (1100, 583)],
# [(801, 399), (1245, 346), (949, 179), (1100, 582)],
# [(800, 397), (1247, 345), (949, 178), (1100, 581)],
# [(800, 395), (1249, 343), (950, 176), (1102, 580)],
# [(800, 394), (1249, 342), (950, 174), (1101, 579)],
# [(799, 392), (1251, 341), (951, 172), (1101, 578)],
# [(798, 390), (1249, 339), (950, 169), (1101, 576)],
# [(796, 389), (1250, 337), (948, 168), (1100, 575)],
# [(795, 388), (1248, 335), (948, 165), (1099, 574)],
# [(793, 385), (1248, 332), (946, 163), (1098, 573)],
# [(792, 382), (1248, 330), (945, 160), (1098, 570)],
# [(791, 381), (1248, 328), (945, 157), (1098, 569)],
# [(790, 379), (1248, 326), (944, 155), (1097, 568)],
# [(789, 376), (1249, 324), (944, 152), (1097, 565)],
# [(790, 375), (1247, 320), (943, 148), (1095, 563)],
# [(788, 372), (1247, 318), (942, 146), (1096, 561)],
# [(788, 370), (1249, 316), (941, 143), (1097, 559)],
# [(786, 367), (1248, 313), (941, 140), (1096, 557)],
# [(785, 365), (1247, 310), (939, 137), (1094, 555)],
# [(784, 364), (1248, 309), (937, 133), (1095, 554)],
# [(784, 364), (1250, 308), (938, 132), (1096, 555)],
# [(783, 362), (1250, 306), (940, 131), (1096, 553)],
# [(782, 360), (1252, 306), (937, 128), (1096, 553)],
# [(782, 359), (1251, 303), (926, 120), (1095, 552)],
# [(781, 356), (1253, 302), (939, 124), (1097, 550)],
# [(778, 353), (1253, 299), (938, 121), (1095, 547)],
# [(778, 351), (1252, 296), (935, 117), (1094, 546)],
# [(777, 348), (1253, 293), (931, 111), (1095, 543)],
# [(776, 346), (1253, 291), (936, 110), (1095, 541)],
# [(778, 345), (1256, 289), (937, 109), (1097, 540)],
# [(776, 343), (1256, 287), (935, 104), (1096, 539)],
# [(776, 341), (1256, 284), (935, 103), (1097, 538)],
# [(776, 340), (1257, 284), (934, 101), (1097, 536)],
# [(772, 338), (1255, 281), (932, 99), (1094, 535)],
# [(773, 337), (1255, 279), (932, 97), (1096, 536)],
# [(772, 339), (1254, 280), (931, 98), (1094, 537)],
# [(768, 341), (1252, 284), (929, 101), (1093, 540)],
# [(768, 343), (1252, 284), (927, 102), (1093, 542)],
# [(766, 344), (1251, 286), (925, 103), (1093, 541)],
# [(763, 346), (1250, 288), (923, 104), (1090, 546)],
# [(761, 351), (1248, 292), (922, 108), (1088, 552)],
# [(759, 353), (1245, 295), (919, 111), (1088, 551)],
# [(754, 354), (1245, 292), (915, 111), (1086, 551)],
# [(752, 354), (1245, 290), (915, 109), (1085, 549)],
# [(751, 351), (1244, 287), (914, 106), (1085, 548)],
# [(750, 351), (1243, 286), (911, 105), (1085, 548)],
# [(750, 350), (1244, 285), (913, 104), (1085, 547)],
# [(750, 350), (1243, 283), (910, 101), (1085, 547)],
# [(751, 348), (1244, 282), (911, 100), (1087, 547)],
# [(752, 346), (1245, 279), (911, 98), (1088, 545)],
# [(751, 345), (1246, 277), (911, 96), (1087, 543)],
# [(751, 343), (1246, 276), (910, 93), (1086, 542)],
# [(750, 343), (1246, 274), (910, 92), (1087, 542)],
# [(748, 341), (1246, 273), (909, 90), (1087, 541)],
# [(746, 338), (1245, 270), (907, 87), (1087, 535)],
# [(745, 337), (1246, 268), (907, 85), (1088, 534)],
# [(745, 334), (1245, 265), (905, 82), (1085, 535)],
# [(743, 332), (1245, 263), (904, 78), (1084, 533)],
# [(741, 330), (1245, 261), (905, 77), (1085, 533)],
# [(741, 328), (1245, 260), (903, 74), (1086, 528)],
# [(739, 327), (1245, 259), (902, 73), (1086, 528)],
# [(738, 327), (1245, 258), (901, 72), (1086, 528)],
# [(739, 326), (1247, 256), (902, 70), (1087, 527)],
# [(735, 330), (1248, 254), (901, 70), (1087, 527)],
# [(735, 328), (1249, 252), (901, 68), (1087, 526)],
# [(736, 328), (1246, 250), (902, 67), (1087, 526)],
# [(738, 328), (1247, 248), (901, 66), (1088, 525)],
# [(735, 326), (1248, 247), (898, 63), (1088, 525)],
# [(734, 326), (1247, 247), (897, 63), (1087, 524)],
# [(733, 324), (1247, 245), (896, 60), (1086, 523)],
# [(733, 322), (1248, 243), (896, 57), (1087, 522)],
# [(732, 320), (1249, 240), (896, 55), (1087, 520)],
# [(732, 316), (1249, 237), (895, 52), (1089, 518)],
# [(732, 313), (1251, 232), (897, 46), (1089, 515)],
# [(731, 311), (1251, 230), (890, 41), (1090, 514)],
# [(731, 309), (1254, 229), (895, 41), (1087, 511)],
# [(729, 307), (1252, 225), (894, 38), (1087, 510)],
# [(728, 305), (1253, 223), (893, 35), (1089, 509)],
# [(728, 303), (1253, 222), (895, 34), (1089, 509)],
# [(726, 301), (1253, 218), (895, 31), (1089, 506)],
# [(725, 300), (1254, 218), (892, 27), (1088, 505)],
# [(724, 297), (1254, 216), (891, 26), (1087, 502)],
# [(722, 295), (1252, 212), (890, 22), (1087, 501)],
# [(721, 293), (1252, 210), (889, 19), (1085, 499)],
# [(720, 292), (1252, 209), (888, 18), (1085, 499)],
# [(719, 291), (1252, 208), (889, 17), (1085, 499)],
# [(719, 290), (1253, 207), (887, 15), (1086, 498)],
# [(718, 288), (1253, 205), (887, 14), (1085, 497)],
# [(718, 284), (1254, 200), (888, 8), (1086, 492)],
# [(716, 283), (1255, 198), (887, 6), (1087, 492)]]

# Intrinsic Matrix of the Camera as given in the question
K = np.array([[1.38e+03, 0, 9.46e+02], [0, 1.38e+03, 5.27e+02], [0, 0, 1]])

# Setting World frame coordinates for the paper
world = np.float32([[0,0], [0.216,0], [0.216,0.279], [0,0.279]])

# Computing homography matrix for each frame w.r.t world coordinates
rotation = []
translation = []
for frame_corners in range(len(corners)):
    points = np.float32(corners[frame_corners])
    H = homography(points, world)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(U, Vt)
    t = H[:, 2] / S[0]
    R = rotations(R)
    rotation.append(R)
    translation.append(t)

time = np.arange(147)
# Plotting roll, pitch, yaw vs time
fig, ax = plt.subplots()
ax.plot(time, [r[0] for r in rotation], label = 'Roll')
ax.plot(time, [r[1] for r in rotation], label = 'Pitch')
ax.plot(time, [r[2] for r in rotation], label = 'Yaw')
ax.set_title('Rotation')
plt.legend()
plt.show()

# Plotting Tx, Ty, Tz vs time
fig, bx = plt.subplots()
bx.plot(time, [t[0] for t in translation], label = 'Tx')
bx.plot(time, [t[1] for t in translation], label = 'Ty')
bx.plot(time, [t[2] for t in translation], label = 'Tz')
bx.set_title('Translation')
plt.legend()
plt.show()