'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    plt.waitforbuttonpress(0)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    plt.waitforbuttonpress(0)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    angle_rad =  angle_rad - 2*np.pi*np.floor((angle_rad + np.pi)/(2*np.pi)) 

    return angle_rad


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''
    x, y, theta = init_pose
    k = init_measure.shape[0] // 2
    b = init_measure[0::2]
    r = init_measure[1::2]


    landmark = np.zeros((2 * k, 1))
    landmark[0::2] = x + r*np.cos(warp2pi(theta + b))
    landmark[1::2] = y + r*np.sin(warp2pi(theta + b))

    landmark_cov = np.zeros((2 * k, 2 * k))

    for i in range(k):
        # Jacobian of the landmark position w.r.t the measurement (beta, range)
        del_cos = np.cos(warp2pi(theta + b[i])).item() 
        del_sin = np.sin(warp2pi(theta + b[i])).item()
        r_i = r[i].item()
        
        # Jacobian of landmark wrt measurement (beta, range)
        J_m = np.array(([-r_i*del_sin, del_cos], 
                        [ r_i*del_cos, del_sin]))

        # Jacobian of the landmark position w.r.t  pose (x, y, theta)
        J_p = np.array([[1, 0, -r_i*del_sin], 
                        [0, 1,  r_i*del_cos]])
        
        # Landmark covariance
        landmark_cov[2*i:2*i+2, 2*i:2*i+2] = J_m @ init_measure_cov @ J_m.T + J_p @ init_pose_cov @ J_p.T


    print("Initial Landmark Covariance ")
    print(landmark_cov)
    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''
    theta = X[2, 0]
    d, alpha = control[0,0], control[1, 0]

    # Predicted Pose
    X[0, 0] += d*np.cos(theta)
    X[1, 0] += d*np.sin(theta)
    X[2, 0]  = warp2pi(theta + alpha)

    G = np.eye(3 + 2*k)
    G[0, 2] = -d*np.sin(theta)
    G[1, 2] =  d*np.cos(theta)


    # According to the book it can Identity
    
    # F = np.eye(3)
    # According to the local frame of error
    F = np.array([[np.cos(theta), -np.sin(theta), 0], 
                  [np.sin(theta),  np.cos(theta), 0], 
                  [     0       ,       0       , 1]])

    # Predicted Covariance
    P = G @ P @ G.T 
    P[0:3, 0:3] += F @ control_cov @ F.T

    return X, P


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''

    x, y, theta = X_pre[0,0], X_pre[1,0], X_pre[2, 0]
    landmarks = X_pre[3:]
    # beta, range
    b = measure[0::2]
    r = measure[1::2]

    # expected beta, range
    measure_expected = np.zeros((2*k, 1))
    lx, ly = landmarks[0::2], landmarks[1::2]
    measure_expected[0::2] = warp2pi(np.arctan2((ly - y), (lx - x)) - theta) 
    measure_expected[1::2] = np.sqrt((lx-x)**2 + (ly - y)**2)


    # Landmarks in global frame
    global_landmarks = np.zeros((2*k, 1))
    global_landmarks[0::2] = x + r*np.cos(warp2pi(theta + b))
    global_landmarks[1::2] = y + r*np.sin(warp2pi(theta + b))

    # Measurement Covariance Q
    Q = np.zeros((2*k, 2*k))
    # Jacobian H
    H = np.zeros((2*k, 2*k + 3))

    for i in range(k):
        lx, ly = global_landmarks[2*i].item() , global_landmarks[2*i + 1].item()
        qt = (lx -x)**2 + (ly - y)**2
        
        

        H_P = np.array([[(ly -  y)/qt         , (x - lx )/qt         , -1],
                        [(x  - lx)/np.sqrt(qt), (y  - ly)/np.sqrt(qt),  0]])


        H_L = np.array([[(y  - ly)/qt         , (lx - x)/qt         ],
                        [(lx - x )/np.sqrt(qt), (ly - y)/np.sqrt(qt)]])


        H[2*i:2*i + 2, 0:3] = H_P
        H[2*i:2*i + 2, 3 + 2*i: 3 + 2*i + 2] = H_L

        Q[2*i:2*i + 2, 2*i:2*i + 2] = measure_cov

    # Kalman Gain
    K = P_pre @ H.T @ np.linalg.inv(H @ P_pre @ H.T + Q)

    # Updated Mean & Covariance
    X_pre += K @ (measure - measure_expected)
    P_pre  = (np.eye(3 + 2*k) - K @ H) @ P_pre
    
    return X_pre, P_pre


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    plt.title('EKF SLAM: Ground Truth vs Estimated Landmarks')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    
    # True Measurement
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    plt.scatter(l_true[0::2], l_true[1::2], c='r', label='Ground Truth')
    
    # Estimated Measurement
    l_estimated = X[3:]
    plt.scatter(l_estimated[0::2], l_estimated[1::2], c='b', label='Estimated')

    
    # Exit to get the distances
    plt.axis('equal')
    plt.legend()
    plt.pause(0)
    
    
    l_true = l_true.reshape(-1, 2)
    l_estimated = l_estimated.reshape(-1, 2)
    
    for i in range(k):
        l_cov = P[3 + 2*i:3 + 2*i+ 2, 3 + 2*i:3 + 2*i + 2]
        error = (l_true[i] - l_estimated[i])

        # Euclidean Distance
        euclidean_distance = np.linalg.norm(l_true[i] - l_estimated[i])
        error = error.flatten()

        # Mahalanobis Distance
        mahalanobis_distance = np.sqrt(error.T @ np.linalg.inv(l_cov) @ error)
        
        print(f"Landmark :{i+1}")
        print(f"Euclidean distance: {euclidean_distance}")
        print(f"Mahalanobis distance: {mahalanobis_distance}")

    print("Final Pose and Landmark Covariances")
    print(P)


def main():
    # TEST: Setup uncertainty parameters
    sig_x = 0.25
    sig_y = 0.1
    sig_alpha = 0.1
    sig_beta = 0.01
    sig_r = 0.08


    # Generate variance from standard deviation
    sig_x2 = sig_x**2
    sig_y2 = sig_y**2
    sig_alpha2 = sig_alpha**2
    sig_beta2 = sig_beta**2
    sig_r2 = sig_r**2

    # Open data file and read the initial measurements
    data_file = open("../data/data.txt")
    line = data_file.readline()
    fields = re.split('[\t ]', line)[:-1]
    arr = np.array([float(field) for field in fields])
    measure = np.expand_dims(arr, axis=1)
    t = 1

    # Setup control and measurement covariance
    control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
    measure_cov = np.diag([sig_beta2, sig_r2])

    # Setup the initial pose vector and pose uncertainty
    pose = np.zeros((3, 1))
    pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

    ##########
    # TODO: initialize landmarks
    k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                               pose_cov)

    # Setup state vector X by stacking pose and landmark states
    # Setup covariance matrix P by expanding pose and landmark covariances
    X = np.vstack((pose, landmark))
    P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                  [np.zeros((2 * k, 3)), landmark_cov]])

    # Plot initial state and covariance
    last_X = X
    draw_traj_and_map(X, last_X, P, 0)

    # Core loop: sequentially process controls and measurements
    for line in data_file:
        fields = re.split('[\t ]', line)[:-1]
        arr = np.array([float(field) for field in fields])

        # Control
        if arr.shape[0] == 2:
            print(f'{t}: Predict step')
            d, alpha = arr[0], arr[1]
            control = np.array([[d], [alpha]])

            ##########
            # TODO: predict step in EKF SLAM
            X_pre, P_pre = predict(X, P, control, control_cov, k)

            draw_traj_and_pred(X_pre, P_pre)

        # Measurement
        else:
            print(f'{t}: Update step')
            measure = np.expand_dims(arr, axis=1)

            ##########
            # TODO: update step in EKF SLAM
            X, P = update(X_pre, P_pre, measure, measure_cov, k)

            draw_traj_and_map(X, last_X, P, t)
            last_X = X
            t += 1

    # EVAL: Plot ground truth landmarks and analyze distances
    evaluate(X, P, k)


if __name__ == "__main__":
    main()
