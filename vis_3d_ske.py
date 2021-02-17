from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import os
import os.path as osp
import numpy as np
import cv2
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

# each joint is connected to some other joint:
connections = [1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10, 0,
               12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11]
lines = zip(range(26), connections)
lines.remove((1, 0))


def draw_frames(ax, frame, ax_lim):
    x, y, z = [frame[:, i] for i in range(3)]

    # draw joints
    ax.scatter(x, z, y, c='r', marker='o', s=15)
    # draw connections between joints
    for i, j in lines:
        ax.plot([x[i], x[j]], [z[i], z[j]], [y[i], y[j]], c='g', linewidth=2)

    set_axes_equal(ax, ax_lim)


def get_axes_lim(joints):
    joints = joints.reshape(-1, 3)

    ax_min = [joints[:, i].min() for i in range(3)]
    ax_max = [joints[:, i].max() for i in range(3)]
    ax_mid = [(ax_max[i] + ax_min[i]) / 2.0 for i in range(3)]
    max_range = np.array([ax_max[i] - ax_min[i] for i in range(3)]).max() / 2.0

    ax_lim = [(ax_mid[i] - max_range, ax_mid[i] + max_range) for i in range(3)]

    return ax_lim


def set_axes_equal(ax, ax_lim):
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.grid(False)

    ax.set_xlim(ax_lim[0][0], ax_lim[0][1])
    ax.set_ylim(ax_lim[2][0], ax_lim[2][1])
    ax.set_zlim(ax_lim[1][0], ax_lim[1][1])

    x_lim = ax.get_xlim3d()
    y_lim = ax.get_ylim3d()
    z_lim = ax.get_zlim3d()

    ax_range = [abs(lim[1] - lim[0]) for lim in [x_lim, y_lim, z_lim]]
    radius = np.array(ax_range).max()
    ax_mid = [(lim[0] + lim[1]) * 0.5 for lim in [x_lim, y_lim, z_lim]]
    ax_lim = [(ax_mid[i] - radius, ax_mid[i] + radius) for i in range(3)]

    ax.set_xlim3d(ax_lim[0][0], ax_lim[0][1])
    ax.set_ylim3d(ax_lim[1][0], ax_lim[1][1])
    ax.set_zlim3d(ax_lim[2][0], ax_lim[2][1])


def draw_ske_data(ske_data, ske_name,save_path, make_video=False):
    joints = purge_ske(ske_data)
    ax_lim = get_axes_lim(joints)
    joints = joints.reshape(-1, 25, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.tight_layout()
    ax.set_aspect('equal')
    ax.view_init(12, -81)

    save_path = osp.join(save_path, ske_name)
    if not osp.exists(save_path):
        os.mkdir(save_path)

    for i in range(joints.shape[0]):
        draw_frames(ax, joints[i], ax_lim)
        # plt.draw()
        plt.savefig(osp.join(save_path, 'f-%d.png') % i)
        # plt.pause(.5)
        ax.clear()

    if make_video:
        make_3d_ske_avi(ske_name, save_path, joints.shape[0])


def make_3d_ske_avi(ske_name, save_path, length):
    assert osp.exists(save_path)

    # rgb_file = osp.join(rgb_path, ske_name + '_rgb.avi')
    ske_3d_file = osp.join(save_path, ske_name + '_3d.avi')

    # assert osp.exists(rgb_file), 'Error: RGB video file %s not found' % rgb_file
    # try:
    #     rgb_video = cv2.VideoCapture(rgb_file)
    # except Exception, e:
    #     print 'Error: OpenCV failed to read video file %s' % rgb_file
    #     print e.args

    # fps = rgb_video.get(cv2.CAP_PROP_FPS)
    # size = (int(rgb_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #         int(rgb_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    img = cv2.imread(osp.join(save_path, 'f-0.png'))
    fps = 30
    height, width = img.shape[:2]
    ske_3d_video = cv2.VideoWriter(ske_3d_file, cv2.VideoWriter_fourcc(*'DIVX'),
                                   fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX  # text font to show text
    f_pos = (width - 40, height - 40)  # bottom-left corner of the frame index shown in the image.
    text_color = (0, 0, 255)  # frame index color (red)

    for i in range(length):
        img = cv2.imread(osp.join(save_path, 'f-%d.png' % i))
        img = cv2.putText(img, str(i), f_pos, font, 0.5, text_color, 1, cv2.LINE_AA)
        ske_3d_video.write(img)

    ske_3d_video.release()


# Utility functions to find a suitable view point
def add_azim(ax, plt, angle):
    azm = ax.axes.azim
    ele = ax.axes.elev
    ax.view_init(ele, azm + angle)
    plt.draw()


def sub_azim(ax, plt, angle):
    azm = ax.axes.azim
    ele = ax.axes.elev
    ax.view_init(ele, azm - angle)
    plt.draw()


def add_elev(ax, plt, angle):
    azm = ax.axes.azim
    ele = ax.axes.elev
    ax.view_init(ele + angle, azm)
    plt.draw()


def sub_elev(ax, plt, angle):
    azm = ax.axes.azim
    ele = ax.axes.elev
    ax.view_init(ele - angle, azm)
    plt.draw()


def reset_view(ax, plt):
    ax.view_init(0, 0)
    plt.draw()

def purge_ske(ske_joint):
    zero_row = []
    for i in range(len(ske_joint)):
        if (ske_joint[i, :] == np.zeros((1, 150))).all():
            zero_row.append(i)
    ske_joint = np.delete(ske_joint, zero_row, axis=0)
    if (ske_joint[:, 0:75] == np.zeros((ske_joint.shape[0], 75))).all():
        ske_joint = np.delete(ske_joint, range(75), axis=1)
    elif (ske_joint[:, 75:150] == np.zeros((ske_joint.shape[0], 75))).all():
        ske_joint = np.delete(ske_joint, range(75, 150), axis=1)
    return ske_joint

def main(ske_data, ske_name, save_path):

    # skeleton data is T*(25*3)
    # ske_name: you can define it, such as 'kicking'
    # save path: where you save output, such as './' 

    draw_ske_data(ske_data, ske_name, save_path, make_video=True)


if __name__ == '__main__':
    main()
