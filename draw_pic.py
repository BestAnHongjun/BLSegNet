import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('font', family='Times New Roman')
import mpl_toolkits.axisartist as axisartist


if __name__ == "__main__":
    # BLSeg
    BLSeg_fps = [71.77, 54.79, 26.52, 19.28]
    BLSeg_miou_cnsoftbei = [93.57, 93.56, 94.90, 95.07]
    BLSeg_miou_craic = [98.26, 98.05, 98.34, 98.53]
    BLSeg_miou_raicom = [98.73, 98.58, 98.70, 98.72]

    # PPLiteSeg
    PPLiteSeg_fps = [22.20, 14.05]
    PPLiteSeg_miou_cnsoftbei = [90.34, 91.27]
    PPLiteSeg_miou_craic = [98.04, 98.47]
    PPLiteSeg_miou_raicom = [95.77, 98.69]

    # BiSeNet
    BiSeNet_fps = [16.27]
    BiSeNet_miou_cnsoftbei = [97.08]
    BiSeNet_miou_craic = [97.98]
    BiSeNet_miou_raicom = [97.74]

    # 创建画布
    fig = plt.figure()
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis["bottom"].set_axisline_style("-|>", size=1.5)
    ax.axis["left"].set_axisline_style("-|>", size=1.5)
    ax.axis["top"].set_visible(False)
    ax.axis["right"].set_visible(False)

    # 传入x,y，通过plot画图,并设置折线颜色、透明度、折线样式和折线宽度  标记点、标记点大小、标记点边颜色、标记点边宽
    plt.plot(np.array(BLSeg_fps), BLSeg_miou_craic, color='red', linestyle='--', linewidth=2, marker='^'
             , markeredgecolor='red', markersize='10', label=r"BLSegNet$ \bf (ours)$")
    plt.plot(np.array(PPLiteSeg_fps), PPLiteSeg_miou_craic, color='blue', linestyle='--', linewidth=2, marker='p'
             , markeredgecolor='blue', markersize='10', label=r"PPLiteSeg")
    plt.plot(np.array(BiSeNet_fps), BiSeNet_miou_craic, color='green', linestyle='--', linewidth=2, marker='*'
             , markeredgecolor='green', markersize='10', label=r"BiSeNet V2")
    plt.xlabel('Inference Speed(FPS)')
    plt.ylabel('TrackingLine-mIoU(%)')
    plt.xlim(10, 80.1)
    plt.ylim(97.9, 98.6)

    plt.text(16.27 + 2, 97.98 - 0.008, "BiSeNet V2")
    for x, y, t in zip(PPLiteSeg_fps, PPLiteSeg_miou_craic, ["PPLiteSeg(STDC1)", "PPLiteSeg(STDC2)"]):
        plt.text(x + 2, y - 0.008, t)
    for x, y, t in zip(BLSeg_fps[1:], BLSeg_miou_craic[1:], [r"BLSegNet-nano$\bf (ours)$", r"BLSegNet-mini$\bf (ours)$", r"BLSegNet-tiny$\bf (ours)$", r"BLSegNet-std$\bf (ours)$"][1:]):
        plt.text(x + 2, y - 0.008, t)
    plt.text(BLSeg_fps[0] - 10, BLSeg_miou_craic[0] + 0.02, r"BLSegNet-nano$\bf (ours)$")

    plt.legend()
    plt.grid(True, which='both', ls='dashed')
    plt.show()
