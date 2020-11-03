import cv2
import numpy as np
import scipy.io as sio
from scipy.linalg import null_space
import matplotlib.pyplot as plt
import sys
from sklearn.neighbors import NearestNeighbors
from mpl_toolkits.mplot3d import Axes3D

def compute_dsift(img, stride, size):
    sift = cv2.xfeatures2d.SIFT_create()
    kp = [cv2.KeyPoint(x,y,size) for y in range(img.shape[0])
                                 for x in range(img.shape[1])]
    kp,dense_feature = sift.compute(img,kp)
    return dense_feature

def find_match(img1, img2):
    img1 = img_left
    img2 = img_right
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=.02)
    kp1,des1 = sift.detectAndCompute(img1,None)
    kp2,des2 = sift.detectAndCompute(img2,None)
    neigh1 = NearestNeighbors(n_neighbors=2)
    neigh1.fit(des2)
    match1 = neigh1.kneighbors(des1) #[0]-distance [1]-index
    x1=[]
    x2=[]
    for i in range(np.size(match1[0],0)):
        if match1[0][i][0] < 0.7*match1[0][i][1]:
            x2.append(kp2[match1[1][i][0]].pt)
            x1.append(kp1[i].pt)
    x1 = np.floor(np.matrix(x1))
    x2 = np.floor(np.matrix(x2))

    neigh2 =  NearestNeighbors(n_neighbors=2)
    neigh2.fit(des1)
    match2 = neigh2.kneighbors(des2)
    xx1=[]
    xx2=[]
    for i in range(np.size(match2[0],0)):
        if match2[0][i][0] < 0.7*match2[0][i][1]:
            xx1.append(kp1[match2[1][i][0]].pt)
            xx2.append(kp2[i].pt)
    xx1 = np.floor(np.matrix(xx1))
    xx2 = np.floor(np.matrix(xx2))
    X1 = []
    X2 = []
    for ii in range(np.size(x1,0)):
        for jj in range(np.size(xx1,0)):
            if (x1[ii,:] == xx1[jj,:]).all():
                X1.append(xx1[jj,:])
                X2.append(xx2[jj,:])
                x1[ii,:]=[[0,0]]
    pts1 = np.reshape(np.array(X1),[np.size(X1,0),2])
    pts2 = np.reshape(np.array(X2),[np.size(X2,0),2])
    return pts1, pts2

def compute_F(pts1, pts2):
    ransac_iter = 100000
    ransac_thr = 0.1
    max_count = 0
    iter = 0

    pts1_3 = np.ones((np.size(pts1,0),3))
    pts1_3[:,0:2] = pts1
    pts2_3 = np.ones((np.size(pts2,0),3))
    pts2_3[:,0:2] = pts2

    while(iter<=ransac_iter):
        iter += 1
        random_i = np.random.choice(range(np.size(pts1,0)),8,replace=False)
        A = np.ones((8,9))
        U_X = pts1[random_i,0]
        U_Y = pts1[random_i,1]
        V_X = pts2[random_i,0]
        V_Y = pts2[random_i,1]

        A[:,0] = U_X*V_X
        A[:,1] = U_Y*V_X
        A[:,2] = V_X
        A[:,3] = U_X*V_Y
        A[:,4] = U_Y*V_Y
        A[:,5] = V_Y
        A[:,6] = U_X
        A[:,7] = U_Y

        U,D,V_T = np.linalg.svd(A)
        V = np.transpose(V_T)/V_T[-1,-1]
        F_temp = np.reshape(V[:,-1],[3,3])
        FU, FD, FV = np.linalg.svd(F_temp)
        # Clean-up
        FD_ = np.zeros((3,3))
        FD_[0,0] = FD[0]
        FD_[1,1] = FD[1]

        F_temp_ = np.matmul(np.matmul(FU,FD_),FV)
        e_line = np.transpose(np.matmul(F_temp_, np.transpose(pts1_3)))
        temp = e_line*pts2_3
        dist = np.absolute(temp[:,0] + temp[:,1] + temp[:,2])/((e_line[:,0]**2+e_line[:,1]**2)**0.5)
        count = sum(dist<ransac_thr)
        if count>max_count:
            max_count = count
            F = F_temp_
    return F

def triangulation(P1, P2, pts1, pts2):
    pts3D = np.zeros((np.size(pts1,0),3))
    pts1_3 = np.ones((np.size(pts1,0),3))
    pts2_3 = np.ones((np.size(pts2,0),3))
    pts1_3[:,0:2] = pts1
    pts2_3[:,0:2] = pts2

    pts1_3x = np.zeros((np.size(pts1,0),3,3))
    pts2_3x = np.zeros((np.size(pts2,0),3,3))
    pts1_3x[:,0,1] = -pts1_3[:,2]
    pts1_3x[:,0,2] = pts1_3[:,1]
    pts1_3x[:,1,0] = pts1_3[:,2]
    pts1_3x[:,1,2] = -pts1_3[:,0]
    pts1_3x[:,2,0] = -pts1_3[:,1]
    pts1_3x[:,2,1] = pts1_3[:,0]

    pts2_3x[:,0,1] = -pts2_3[:,2]
    pts2_3x[:,0,2] = pts2_3[:,1]
    pts2_3x[:,1,0] = pts2_3[:,2]
    pts2_3x[:,1,2] = -pts2_3[:,0]
    pts2_3x[:,2,0] = -pts2_3[:,1]
    pts2_3x[:,2,1] = pts2_3[:,0]

    A = np.zeros((np.size(pts1,0),4,4))
    temp1 = np.matmul(pts1_3x, P1)
    temp2 = np.matmul(pts2_3x, P2)
    A[:,0:2,:] = temp1[:,0:2,:]
    A[:,2:4,:] = temp2[:,0:2,:]

    AU,AD,AV = np.linalg.svd(A)

    for i in range(np.size(AV,0)):
        temp = AV[i,-1,0:4]/AV[i,-1,-1]
        pts3D[i,:] = temp[0:3]

    return pts3D

def disambiguate_pose(Rs, Cs, pts3Ds):
    ns = np.zeros((4,1))
    for i in range(4):
        pts3D = pts3Ds[i]
        C = Cs[i]
        R = Rs[i]
        n = sum(np.matmul(R[:,2],np.transpose(pts3D)-C)>0)
        ns[i,0] = n
    ii = np.argmax(ns)
    pts3D = pts3Ds[ii]
    C = Cs[ii]
    R = Rs[ii]
    return R, C, pts3D

def compute_rectification(K, R, C):
    r_x = C/np.linalg.norm(C)
    r_y = np.zeros((3))
    r_y[0] = -C[1,0]/np.linalg.norm(C[0:2,0])
    r_y[1] = C[0,0]/np.linalg.norm(C[0:2,0])

    r_z = np.cross(r_x[:,0], r_y)

    R_rect = np.zeros((3,3))
    R_rect[0,:] = r_x[:,0]
    R_rect[1,:] = r_y
    R_rect[2,:] = r_z
    K_inv = np.linalg.inv(K)
    H1 = np.matmul(np.matmul(K,R_rect),K_inv)
    H2 = np.matmul(np.matmul(np.matmul(K,R_rect),np.transpose(R)),K_inv)
    return H1, H2

def dense_match(img1, img2):
    stride = 1
    size = 5
    H = img1.shape[0]
    W = img1.shape[1]
    dense_feature1 = compute_dsift(img1, stride, size)
    dense_feature2 = compute_dsift(img2,stride,size)
    disparity = np.zeros(img1.shape)
    dense_feature1 = np.reshape(dense_feature1,[H,W,128])
    dense_feature2 = np.reshape(dense_feature2,[H,W,128])

    for i in range(H):
        for j in range(W):
            temp_min = 10000
            temp_i = 0
            
            for k in range(j):
                temp = np.linalg.norm(dense_feature1[i,j] - dense_feature2[i][k])
                
                if(temp<=temp_min):
                    temp_i = j-k
                    temp_min=temp
                    
            disparity[i][j] = temp_i
    return disparity
    
# Provided functions by TA
def compute_camera_pose(F, K):
    E = K.T @ F @ K
    R_1, R_2, t = cv2.decomposeEssentialMat(E)
    # 4 cases
    R1, t1 = R_1, t
    R2, t2 = R_1, -t
    R3, t3 = R_2, t
    R4, t4 = R_2, -t

    Rs = [R1, R2, R3, R4]
    ts = [t1, t2, t3, t4]
    Cs = []
    for i in range(4):
        Cs.append(-Rs[i].T @ ts[i])
    return Rs, Cs

def visualize_img_pair(img1, img2):
    img = np.hstack((img1, img2))
    if img1.ndim == 3:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

def visualize_find_match(img1, img2, pts1, pts2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    img_h = img1.shape[0]
    scale_factor1 = img_h/img1.shape[0]
    scale_factor2 = img_h/img2.shape[0]
    img1_resized = cv2.resize(img1, None, fx=scale_factor1, fy=scale_factor1)
    img2_resized = cv2.resize(img2, None, fx=scale_factor2, fy=scale_factor2)
    pts1 = pts1 * scale_factor1
    pts2 = pts2 * scale_factor2
    pts2[:, 0] += img1_resized.shape[1]
    img = np.hstack((img1_resized, img2_resized))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for i in range(pts1.shape[0]):
        plt.plot([pts1[i, 0], pts2[i, 0]], [pts1[i, 1], pts2[i, 1]], 'b.-', linewidth=0.5, markersize=5)
    plt.axis('off')
    plt.show()

def visualize_epipolar_lines(F, pts1, pts2, img1, img2):
    assert pts1.shape == pts2.shape, 'x1 and x2 should have same shape!'
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

    for i in range(pts1.shape[0]):
        x1, y1 = int(pts1[i][0] + 0.5), int(pts1[i][1] + 0.5)
        ax1.scatter(x1, y1, s=5)
        p1, p2 = find_epipolar_line_end_points(img2, F, (x1, y1))
        ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    for i in range(pts2.shape[0]):
        x2, y2 = int(pts2[i][0] + 0.5), int(pts2[i][1] + 0.5)
        ax2.scatter(x2, y2, s=5)
        p1, p2 = find_epipolar_line_end_points(img1, F.T, (x2, y2))
        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=0.5)

    ax1.axis('off')
    ax2.axis('off')
    plt.show()

def find_epipolar_line_end_points(img, F, p):
    img_width = img.shape[1]
    el = np.dot(F, np.array([p[0], p[1], 1]).reshape(3, 1))
    p1, p2 = (0, -el[2] / el[1]), (img.shape[1], (-img_width * el[0] - el[2]) / el[1])
    _, p1, p2 = cv2.clipLine((0, 0, img.shape[1], img.shape[0]), p1, p2)
    return p1, p2

def visualize_camera_poses(Rs, Cs):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2 = Rs[i], Cs[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1)
        draw_camera(ax, R2, C2)
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()

def visualize_camera_poses_with_pts(Rs, Cs, pts3Ds):
    assert(len(Rs) == len(Cs) == 4)
    fig = plt.figure()
    R1, C1 = np.eye(3), np.zeros((3, 1))
    for i in range(4):
        R2, C2, pts3D = Rs[i], Cs[i], pts3Ds[i]
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        draw_camera(ax, R1, C1, 5)
        draw_camera(ax, R2, C2, 5)
        ax.plot(pts3D[:, 0], pts3D[:, 1], pts3D[:, 2], 'b.')
        set_axes_equal(ax)
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
        ax.set_zlabel('z axis')
        ax.view_init(azim=-90, elev=0)
    fig.tight_layout()
    plt.show()

def draw_camera(ax, R, C, scale=0.2):
    axis_end_points = C + scale * R.T  # (3, 3)
    vertices = C + scale * R.T @ np.array([[1, 1, 1], [-1, 1, 1], [-1, -1, 1], [1, -1, 1]]).T  # (3, 4)
    vertices_ = np.hstack((vertices, vertices[:, :1]))  # (3, 5)

    # draw coordinate system of camera
    ax.plot([C[0], axis_end_points[0, 0]], [C[1], axis_end_points[1, 0]], [C[2], axis_end_points[2, 0]], 'r-')
    ax.plot([C[0], axis_end_points[0, 1]], [C[1], axis_end_points[1, 1]], [C[2], axis_end_points[2, 1]], 'g-')
    ax.plot([C[0], axis_end_points[0, 2]], [C[1], axis_end_points[1, 2]], [C[2], axis_end_points[2, 2]], 'b-')

    # draw square window and lines connecting it to camera center
    ax.plot(vertices_[0, :], vertices_[1, :], vertices_[2, :], 'k-')
    ax.plot([C[0], vertices[0, 0]], [C[1], vertices[1, 0]], [C[2], vertices[2, 0]], 'k-')
    ax.plot([C[0], vertices[0, 1]], [C[1], vertices[1, 1]], [C[2], vertices[2, 1]], 'k-')
    ax.plot([C[0], vertices[0, 2]], [C[1], vertices[1, 2]], [C[2], vertices[2, 2]], 'k-')
    ax.plot([C[0], vertices[0, 3]], [C[1], vertices[1, 3]], [C[2], vertices[2, 3]], 'k-')

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range, x_middle = abs(x_limits[1] - x_limits[0]), np.mean(x_limits)
    y_range, y_middle = abs(y_limits[1] - y_limits[0]), np.mean(y_limits)
    z_range, z_middle = abs(z_limits[1] - z_limits[0]), np.mean(z_limits)

    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def visualize_disparity_map(disparity):
    plt.imshow(disparity, cmap='jet')
    plt.show()


if __name__ == '__main__':
    # read in left and right images as RGB images
    img_left = cv2.imread('./left.bmp', 1)
    img_right = cv2.imread('./right.bmp', 1)
    visualize_img_pair(img_left, img_right)

    # Step 1: find correspondences between image pair
    pts1, pts2 = find_match(img_left, img_right)
    visualize_find_match(img_left, img_right, pts1, pts2)

    # Step 2: compute fundamental matrix
    F = compute_F(pts1, pts2)
    visualize_epipolar_lines(F, pts1, pts2, img_left, img_right)

    # Step 3: computes four sets of camera poses
    K = np.array([[350, 0, 960/2], [0, 350, 540/2], [0, 0, 1]])
    Rs, Cs = compute_camera_pose(F, K)
    visualize_camera_poses(Rs, Cs)

    # Step 4: triangulation
    pts3Ds = []
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    for i in range(len(Rs)):
        P2 = K @ np.hstack((Rs[i], -Rs[i] @ Cs[i]))
        pts3D = triangulation(P1, P2, pts1, pts2)
        pts3Ds.append(pts3D)
    visualize_camera_poses_with_pts(Rs, Cs, pts3Ds)

    # Step 5: disambiguate camera poses
    R, C, pts3D = disambiguate_pose(Rs, Cs, pts3Ds)

    # Step 6: rectification
    H1, H2 = compute_rectification(K, R, C)
    img_left_w = cv2.warpPerspective(img_left, H1, (img_left.shape[1], img_left.shape[0]))
    img_right_w = cv2.warpPerspective(img_right, H2, (img_right.shape[1], img_right.shape[0]))
    visualize_img_pair(img_left_w, img_right_w)

    # Step 7: generate disparity map
    img_left_w = cv2.resize(img_left_w, (int(img_left_w.shape[1] / 2), int(img_left_w.shape[0] / 2)))  # resize image for speed
    img_right_w = cv2.resize(img_right_w, (int(img_right_w.shape[1] / 2), int(img_right_w.shape[0] / 2)))
    img_left_w = cv2.cvtColor(img_left_w, cv2.COLOR_BGR2GRAY)  # convert to gray scale
    img_right_w = cv2.cvtColor(img_right_w, cv2.COLOR_BGR2GRAY)
    disparity = dense_match(img_left_w, img_right_w)
    visualize_disparity_map(disparity)

    # save to mat
    sio.savemat('stereo.mat', mdict={'pts1': pts1, 'pts2': pts2, 'F': F, 'pts3D': pts3D, 'H1': H1, 'H2': H2,
                                     'img_left_w': img_left_w, 'img_right_w': img_right_w, 'disparity': disparity})
