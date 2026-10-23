#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3D 爱心（Anaconda / numpy + matplotlib 版）

与终端字符版不同，这一版生成真正的三维三角网格，用 matplotlib 做
光照着色渲染，可以开窗口实时旋转，也能导出 PNG / GIF / 交互式 HTML。

形状构造：
    1. 二维心形距离场 d(x, y) < 0 表示在心形内部（精确 SDF）
    2. 厚度分布 h(s) = T * sqrt(1 - ((D - s) / D)^2)，s = -d 为离轮廓的深度
    3. 三维隐式场 F(x, y, z) = max(d, |z| - h(s))
       （取 max 保证心形之外严格为正，否则 z=0 平面会退化成等值面）
    4. 在 F 的零等值面上做 marching cubes，得到闭合三角网格

运行（用 Anaconda 的 Python）：
    D:\\Anacanda\\python.exe heart3d_mpl.py                  # 弹窗实时旋转
    D:\\Anacanda\\python.exe heart3d_mpl.py --save heart.png
    D:\\Anacanda\\python.exe heart3d_mpl.py --save heart.gif
    D:\\Anacanda\\python.exe heart3d_mpl.py --html heart.html
"""

import argparse
import os
import sys

import numpy as np

# ---------------------------------------------------------------- 形状参数

THICK = 0.30                                 # 最厚处的半厚度（心形高归一化为 1.0）
BASE_COLOR = np.array([0.94, 0.07, 0.16])    # 心形本色
BG_DEFAULT = "#0d0d14"

# 世界坐标下的固定光源（右上偏前），物体转动时高光会在表面上移动
LIGHT = np.array([-0.45, 0.58, 0.68])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


# ---------------------------------------------------------------- 二维心形

def sd_heart2d(px, py):
    """二维心形距离场（向量化）。内部为负，心形大致占 x∈[-0.6,0.6], y∈[0,1.1]。"""
    px = np.abs(px)
    # 上半部分：两个圆
    r1 = np.sqrt((px - 0.25) ** 2 + (py - 0.75) ** 2) - 0.35355339059327373
    # 下半部分：两段中垂线距离取最小
    d1 = px ** 2 + (py - 1.0) ** 2
    m = np.maximum(px + py, 0.0) * 0.5
    d2 = (px - m) ** 2 + (py - m) ** 2
    r2 = np.sqrt(np.minimum(d1, d2))
    r2 = np.where(px - py >= 0.0, r2, -r2)
    return np.where(py + px > 1.0, r1, r2)


def scan_heart():
    """扫描心形包围盒与最大内深，用于归一化到「高 1.0、居中」。"""
    n = 400
    xs = np.linspace(-0.8, 0.8, n)
    ys = np.linspace(-0.25, 1.3, n)
    X, Y = np.meshgrid(xs, ys)
    D = sd_heart2d(X, Y)
    inside = D < 0.0
    cols = inside.any(axis=0)
    rows = inside.any(axis=1)
    xmin = xs[np.argmax(cols)]
    xmax = xs[len(xs) - 1 - np.argmax(cols[::-1])]
    ymin = ys[np.argmax(rows)]
    ymax = ys[len(ys) - 1 - np.argmax(rows[::-1])]
    return (xmin + xmax) * 0.5, (ymin + ymax) * 0.5, 1.0 / (ymax - ymin), float(-D.min())


_CX, _CY, _SCALE, _DEPTH_RAW = scan_heart()
DEPTH = _DEPTH_RAW * _SCALE


def sd_heart(x, y):
    """归一化后的二维心形距离场：高度 1.0，几何中心在原点。"""
    return sd_heart2d(x / _SCALE + _CX, y / _SCALE + _CY) * _SCALE


# 距离场在心形下半部的对称轴上有折痕（最近的边界点在 V 形两条边之间切换），
# 用「模糊深度 × 贴边收敛」可以抹掉它，但模糊会削弱曲面细节、看起来发硬，
# 权衡后这里直接把深度小幅抬离 0：既能盖住折痕，又保持轮廓处收边。
_EDGE_TAPER = 0.06


def depth2d(x, y):
    """离轮廓深度（≥0）。轮廓处为 0，越往里越大。"""
    return np.maximum(-sd_heart(x, y), 0.0)


def half_thickness(s):
    """距轮廓深度 s 处的半厚度：边缘为 0，中心最厚。"""
    s = np.clip(s, 0.0, DEPTH)
    return THICK * np.sqrt(np.maximum(0.0, 1.0 - ((DEPTH - s) / DEPTH) ** 2))


# ---------------------------------------------------------------- 三维网格

def build_mesh(res=140):
    """在 F = |z| - h(s) 的零等值面上提取三角网格，返回 (verts, faces)。"""
    from skimage import measure

    pad = 0.03
    gx = np.linspace(-0.56 - pad, 0.56 + pad, res)
    gy = np.linspace(-0.50 - pad, 0.50 + pad, res)
    # 轮廓处曲面近乎垂直，z 方向采样足够密才不会出锯齿
    gz = np.linspace(-THICK - pad, THICK + pad, max(32, int(res * 0.75)))

    X, Y = np.meshgrid(gx, gy)                 # (ny, nx)
    d = sd_heart(X, Y)
    h = half_thickness(depth2d(X, Y))

    # F[k, j, i] = max(d, |z_k| - h[j, i])，用广播即可，不必生成三维坐标网格。
    # 取 max 是为了让心形之外严格为正，否则 |z| 在 z=0 平面上会等于 0，
    # marching cubes 会把整块外部平面当成等值面抽出来。
    F = np.maximum(np.abs(gz)[:, None, None] - h[None, :, :], d[None, :, :])

    verts, faces, _, _ = measure.marching_cubes(
        F, level=0.0, spacing=(gz[1] - gz[0], gy[1] - gy[0], gx[1] - gx[0])
    )
    # marching_cubes 的顶点轴顺序与体数据一致：(z, y, x)，换算成 (x, y, z)
    verts = np.column_stack([
        verts[:, 2] + gx[0],
        verts[:, 1] + gy[0],
        verts[:, 0] + gz[0],
    ])
    return verts, faces


def field3d(x, y, z):
    """三维隐式场（向量化）。内部为负。"""
    d = sd_heart(x, y)
    h = half_thickness(depth2d(x, y))
    return np.maximum(np.abs(z) - h, d)


def gradients(x, y, z, eps=1e-3):
    """隐式场的中心差分梯度，方向朝外。"""
    gx = field3d(x + eps, y, z) - field3d(x - eps, y, z)
    gy = field3d(x, y + eps, z) - field3d(x, y - eps, z)
    gz = field3d(x, y, z + eps) - field3d(x, y, z - eps)
    return np.column_stack([gx, gy, gz])


def face_normals(verts, faces):
    """用隐式场在面心处的梯度作为法线。

    比直接用三角形叉积平滑得多：网格在陡峭边缘会有阶梯状三角形，
    几何法线随之抖动，渲染出来是一圈圈水波纹。
    """
    centroids = verts[faces].mean(axis=1)
    n = gradients(centroids[:, 0], centroids[:, 1], centroids[:, 2])
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm[norm == 0.0] = 1.0
    return n / norm


# ---------------------------------------------------------------- 着色

def shaded_colors(normals, angle_deg=0.0, ambient=0.26, spec=0.38):
    """按「跟随相机」的光源做漫反射 + 高光，返回每个面的 RGB。

    normals 必须已经换到绘图坐标（matplotlib 竖轴是 z），即心形的
    (x, y, z) → (x, z, y)。angle_deg 是相机方位角，先用它把法线转回
    相机坐标系，光源就相对屏幕固定，转到任何角度都不会背光发黑。
    """
    # 相机在方位角 azim 处看向原点，世界坐标转相机坐标需绕 z 轴转 (90° - azim)
    a = np.radians(90.0 - angle_deg)
    ca, sa = np.cos(a), np.sin(a)
    nx = normals[:, 0] * ca - normals[:, 1] * sa
    ny = normals[:, 0] * sa + normals[:, 1] * ca
    nz = normals[:, 2]

    diff = np.clip(nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2], 0.0, None)

    # 半程向量（视线从 +y 看向原点）
    hv = LIGHT + np.array([0.0, 1.0, 0.0])
    hv = hv / np.linalg.norm(hv)
    spec_val = np.clip(nx * hv[0] + ny * hv[1] + nz * hv[2], 0.0, None) ** 14

    # 天光：朝上的面稍亮，避免暗部死黑
    sky = 0.5 + 0.5 * nz
    shade = ambient + 0.18 * sky + 0.78 * diff
    shade = np.clip(shade, 0.0, None)

    rgb = BASE_COLOR[None, :] * shade[:, None]
    rgb += spec * spec_val[:, None]
    return np.clip(rgb, 0.0, 1.0)


# ---------------------------------------------------------------- 绘图

def prepare_mesh(verts, faces):
    """把网格转到绘图坐标，并算好每个面的颜色。

    法线取隐式场梯度的方向，天然朝外，不需要再按绕向翻转；
    对闭合曲面用「面心与法线点积的均值」判方向会正负抵消，判不准。
    """
    plot_verts = verts[:, [0, 2, 1]]                    # 心形 y(上) -> 绘图 z
    normals = face_normals(verts, faces)[:, [0, 2, 1]]
    colors = shaded_colors(normals)
    return plot_verts, normals, colors


def apply_colors(coll, colors):
    """同时设置面颜色与描边颜色。

    只给面着色时，相邻三角形之间会留下细缝，密集成片后看起来像布纹；
    用同色描边填缝即可消除。
    """
    coll.set_facecolor(colors)
    coll.set_edgecolor(colors)


def make_figure(plot_verts, faces, colors, elev, bg, size, dpi, antialias=True):
    import matplotlib
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = matplotlib.pyplot.figure(figsize=(size, size), dpi=dpi, facecolor=bg)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg)

    coll = Poly3DCollection(
        plot_verts[faces], facecolors=colors, edgecolors="none",
        linewidths=0.5, antialiased=antialias, shade=False,
    )
    ax.add_collection3d(coll)
    apply_colors(coll, colors)

    lim = 0.60
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=1.22)
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=90)
    ax.set_proj_type("persp", focal_length=0.6)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    return fig, ax, coll


def camera_dir(azim_deg, elev_deg):
    """matplotlib 相机在绘图坐标下的方向单位向量。"""
    a = np.radians(azim_deg)
    e = np.radians(elev_deg)
    return np.array([np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)])


def render_still(plot_verts, faces, normals, path, elev, azim, dpi, bg, size):
    """渲染单帧图像。静态图做背面剔除，能显著减少画家算法排序错误。"""
    import matplotlib.pyplot as plt

    cam = camera_dir(azim, elev)
    keep = (normals @ cam) > 0.0
    colors = shaded_colors(normals, azim)

    fig, ax, coll = make_figure(plot_verts, faces[keep], colors[keep],
                                elev, bg, size, dpi, antialias=True)
    ax.view_init(elev=elev, azim=azim)
    fig.savefig(path, dpi=dpi, facecolor=bg)
    plt.close(fig)


def apply_view(ax, coll, plot_verts, faces, normals, azim, elev):
    """按给定视角剔除背面、更新颜色并移动相机（动画每帧调用）。"""
    keep = (normals @ camera_dir(azim, elev)) > 0.0
    coll.set_verts(plot_verts[faces[keep]])
    apply_colors(coll, shaded_colors(normals, azim)[keep])
    ax.view_init(elev=elev, azim=azim)


def save_animation(plot_verts, faces, normals, path, elev, fps, frames, dpi, bg, size):
    """逐帧渲染后用 PIL 合成 GIF。

    直接交给 matplotlib 的 PillowWriter 时，每帧调色板各自计算，
    红色渐变会被量化成一圈圈色带；这里统一用第一帧的调色板并对
    每帧做抖动（Floyd–Steinberg），过渡就平滑了。
    """
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    fig, ax, coll = make_figure(plot_verts, faces, shaded_colors(normals, 90.0),
                                elev, bg, size, dpi, antialias=True)
    angles = np.linspace(90.0, 450.0, frames, endpoint=False)

    images = []
    for a in angles:
        apply_view(ax, coll, plot_verts, faces, normals, float(a), elev)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor=bg)
        buf.seek(0)
        img = Image.open(buf)
        img.load()
        images.append(img.convert("RGB"))
    plt.close(fig)

    palette = images[0].quantize(colors=255, method=Image.MEDIANCUT)
    quantized = [im.quantize(palette=palette, dither=Image.FLOYDSTEINBERG) for im in images]
    quantized[0].save(
        path, save_all=True, append_images=quantized[1:],
        duration=max(1, int(round(1000.0 / max(fps, 1)))), loop=0, disposal=2,
    )


def save_html(verts, faces, path, bg):
    """plotly 导出：可用鼠标拖拽旋转的交互式 3D 页面。"""
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=np.round(verts[:, 0], 5), y=np.round(verts[:, 1], 5),
                z=np.round(verts[:, 2], 5),
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="#e8172b", opacity=1.0, flatshading=False,
                lighting=dict(ambient=0.30, diffuse=0.85, specular=0.6,
                              roughness=0.3, fresnel=0.15),
                lightposition=dict(x=-1.5, y=2.0, z=2.5),
            )
        ]
    )
    fig.update_layout(
        paper_bgcolor=bg,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=0.0, y=0.0, z=2.6)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        width=820, height=820,
    )
    fig.write_html(path, include_plotlyjs="cdn")


# ---------------------------------------------------------------- 入口

def main():
    parser = argparse.ArgumentParser(
        description="3D 爱心：numpy + matplotlib + marching cubes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--res", type=int, default=120, help="网格分辨率（越大越精细也越慢）")
    parser.add_argument("--save", type=str, default="", help="导出 .png 或 .gif")
    parser.add_argument("--html", type=str, default="", help="导出可拖拽旋转的 HTML")
    parser.add_argument("--no-window", action="store_true", help="不弹窗，只导出")
    parser.add_argument("--fps", type=int, default=15, help="动画帧率")
    parser.add_argument("--frames", type=int, default=60, help="GIF 总帧数")
    parser.add_argument("--dpi", type=int, default=140, help="导出分辨率（GIF 每帧约 1 秒）")
    parser.add_argument("--size", type=float, default=7.0, help="画布边长（英寸）")
    parser.add_argument("--elev", type=float, default=12.0, help="俯视角")
    parser.add_argument("--azim", type=float, default=108.0, help="静态图方位角（90 为正对）")
    parser.add_argument("--bg", type=str, default=BG_DEFAULT, help="背景色")
    args = parser.parse_args()

    headless = bool(args.save or args.html or args.no_window)
    if headless:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  （确保 backend 已确定）

    verts, faces = build_mesh(args.res)
    plot_verts, normals, colors = prepare_mesh(verts, faces)
    print("网格：%d 顶点 / %d 三角面" % (len(verts), len(faces)))

    if args.html:
        save_html(verts, faces, args.html, args.bg)
        print("已导出 HTML：%s" % args.html)

    if args.save:
        ext = os.path.splitext(args.save)[1].lower()
        if ext == ".gif":
            save_animation(plot_verts, faces, normals, args.save, args.elev,
                           args.fps, args.frames, args.dpi, args.bg, args.size)
        else:
            render_still(plot_verts, faces, normals, args.save, args.elev,
                         args.azim, args.dpi, args.bg, args.size)
        print("已导出：%s" % args.save)

    if headless:
        return 0

    # 交互窗口：绕竖轴匀速旋转
    from matplotlib.animation import FuncAnimation

    fig, ax, coll = make_figure(plot_verts, faces, colors, args.elev, args.bg,
                                args.size, args.dpi)
    state = {"azim": 90.0}

    def update(_):
        state["azim"] = (state["azim"] + 2.0) % 360.0
        apply_colors(coll, shaded_colors(normals, state["azim"]))
        ax.view_init(elev=args.elev, azim=state["azim"])
        return (coll,)

    _anim = FuncAnimation(fig, update, interval=1000 // max(args.fps, 1), blit=False)
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
