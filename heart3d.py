#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3D 爱心 —— 纯标准库实现的终端三维渲染器。

形状由「二维心形距离场 + 圆润厚度」构造：

    二维心形  d(x, y)  < 0 表示在心形内部（精确距离场）
    厚度分布  h(s) = T * sqrt(1 - ((D - s) / D)^2)，s = -d 为离边缘的深度
    三维实体  F(x, y, z) = |z| - h(s)

即心形沿 z 轴鼓起：轮廓处厚度收敛为 0（曲面在此垂直收边，接缝平滑），
中心处最厚。再对 F 做光线步进（ray marching），用梯度求法线，
配合漫反射与高光着色，投影成终端字符。

不需要 numpy 或任何第三方库，Python 3.6+ 直接运行。

用法：
    python heart3d.py                 # 旋转动画（Ctrl+C 退出）
    python heart3d.py --static        # 只输出一帧
    python heart3d.py --no-color      # 纯 ASCII，兼容老终端
    python heart3d.py -W 100 -H 48    # 自定义分辨率
    python heart3d.py --zoom 1.3      # 放大
"""

import argparse
import math
import os
import shutil
import sys
import time

# ---------------------------------------------------------------- 心形几何

THICK = 0.30   # 心形最厚处的半厚度
BOUND_R = 0.72  # 包围球半径（用于裁掉射线空程）
MAX_STEPS = 260


def _sd_raw(x, y):
    """二维心形距离场（内部为负）。心形大致占据 x∈[-0.6,0.6], y∈[0,1.1]。"""
    x = abs(x)
    if y + x > 1.0:
        dx = x - 0.25
        dy = y - 0.75
        return math.sqrt(dx * dx + dy * dy) - 0.35355339059327373
    dx1 = x
    dy1 = y - 1.0
    d1 = dx1 * dx1 + dy1 * dy1
    m = (x + y) * 0.5 if (x + y) > 0.0 else 0.0
    dx2 = x - m
    dy2 = y - m
    d2 = dx2 * dx2 + dy2 * dy2
    r = math.sqrt(d1 if d1 < d2 else d2)
    return r if (x - y) >= 0.0 else -r


def _scan_heart():
    """扫描心形包围盒与最大内深，用于归一化到「高 1.0、居中」。"""
    n = 96
    xmin = ymin = 1e9
    xmax = ymax = -1e9
    depth = 0.0
    for j in range(n + 1):
        y = 1.25 - 1.45 * j / float(n)
        for i in range(n + 1):
            x = -0.8 + 1.6 * i / float(n)
            d = _sd_raw(x, y)
            if d < 0.0:
                if x < xmin:
                    xmin = x
                if x > xmax:
                    xmax = x
                if y < ymin:
                    ymin = y
                if y > ymax:
                    ymax = y
                if -d > depth:
                    depth = -d
    return (xmin + xmax) * 0.5, (ymin + ymax) * 0.5, 1.0 / (ymax - ymin), depth


_CX, _CY, _SCALE, _DEPTH_RAW = _scan_heart()

DEPTH = _DEPTH_RAW * _SCALE  # 归一化后的最大内深（世界单位）


def sd_heart(x, y):
    """归一化后的心形距离场：高度 1.0，几何中心在原点，内部为负。"""
    return _sd_raw(x / _SCALE + _CX, y / _SCALE + _CY) * _SCALE


def half_thickness(s):
    """距轮廓深度 s 处的半厚度：边缘为 0，中心为 THICK。"""
    if s <= 0.0:
        return 0.0
    if s >= DEPTH:
        s = DEPTH
    t = (DEPTH - s) / DEPTH
    v = 1.0 - t * t
    if v <= 0.0:
        return 0.0
    return THICK * math.sqrt(v)


def field(x, y, z):
    """三维隐式场。返回 < 0 表示在实体内部。"""
    d = sd_heart(x, y)
    if d >= 0.0:
        return d
    az = z if z >= 0.0 else -z
    return az - half_thickness(-d)


def normal_at(x, y, z, eps=2e-3):
    """中心差分求梯度，作为曲面法线（未归一化）。"""
    return (
        field(x + eps, y, z) - field(x - eps, y, z),
        field(x, y + eps, z) - field(x, y - eps, z),
        field(x, y, z + eps) - field(x, y, z - eps),
    )


def march(ox, oy, oz, dx, dy, dz):
    """沿射线步进求首个交点距离；无交点返回 None。

    先用包围球裁掉相机到物体之间的空程，再用带安全系数的球面步进
    （场的正值作为到表面的距离下界），命中后二分细化。
    """
    # 射线与包围球求交
    b = ox * dx + oy * dy + oz * dz
    c = ox * ox + oy * oy + oz * oz - BOUND_R * BOUND_R
    disc = b * b - c
    if disc <= 0.0:
        return None
    root = math.sqrt(disc)
    t0 = -b - root
    t1 = -b + root
    if t1 <= 0.0:
        return None

    t = t0 if t0 > 0.0 else 0.0
    f = field(ox + dx * t, oy + dy * t, oz + dz * t)
    if f < 0.0:
        return t

    for _ in range(MAX_STEPS):
        # 正值场可视为到表面的距离下界，乘安全系数推进
        step = 0.55 * f
        if step < 0.005:
            step = 0.005
        elif step > 0.12:
            step = 0.12
        t_prev = t
        t += step
        if t > t1:
            return None
        px = ox + dx * t
        py = oy + dy * t
        pz = oz + dz * t
        f = field(px, py, pz)
        if f < 0.0:
            lo = t_prev
            hi = t
            for _ in range(10):
                mid = 0.5 * (lo + hi)
                if field(ox + dx * mid, oy + dy * mid, oz + dz * mid) < 0.0:
                    hi = mid
                else:
                    lo = mid
            return 0.5 * (lo + hi)
    return None


# ---------------------------------------------------------------- 着色

RAMP = " .,:;!*w#%@"

PALETTE = [
    (58, 6, 10), (92, 10, 16), (126, 14, 22), (160, 20, 28),
    (194, 26, 34), (222, 34, 44), (240, 52, 58), (248, 74, 72),
    (252, 100, 92), (254, 130, 116), (255, 165, 148), (255, 205, 190),
]


def shade(nx, ny, nz, vx, vy, vz):
    """漫反射 + 高光，返回 0..1 亮度。"""
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    if ln == 0.0:
        return 0.0
    nx /= ln
    ny /= ln
    nz /= ln

    lx, ly, lz = -0.42, 0.60, 0.68
    diff = nx * lx + ny * ly + nz * lz
    if diff < 0.0:
        diff = 0.0

    hx = lx + vx
    hy = ly + vy
    hz = lz + vz
    hl = math.sqrt(hx * hx + hy * hy + hz * hz)
    spec = 0.0
    if hl > 0.0:
        s = (nx * hx + ny * hy + nz * hz) / hl
        if s > 0.0:
            spec = s ** 30.0

    # 天光：朝上的面亮一些，暗部也能保留形状起伏
    sky = 0.5 + 0.5 * ny
    # 补光：从右下方向的弱光，避免背光面塌成一片死黑
    fill = nx * 0.62 + ny * -0.32 + nz * 0.72
    if fill < 0.0:
        fill = 0.0
    val = 0.05 + 0.26 * sky + 0.60 * diff + 0.24 * fill + 0.36 * spec
    # 掠射面压暗一点，边缘更立体
    facing = 1.0 - abs(nz)
    val -= 0.06 * facing * facing
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


# ---------------------------------------------------------------- 渲染

def render(width, height, angle, color=True, zoom=1.0):
    """渲染一帧，返回若干行字符串。"""
    ca = math.cos(angle)
    sa = math.sin(angle)

    dist = 2.6
    tan_half = 0.25 / zoom
    # 字符高宽比约 1:2，水平方向按终端比例换算
    half_w = tan_half * width / (2.0 * height)

    escapes = []
    if color:
        for r, g, b in PALETTE:
            escapes.append("\x1b[38;2;%d;%d;%dm" % (r, g, b))
    level_count = len(RAMP) - 1
    pal_count = len(PALETTE) - 1

    lines = []
    for j in range(height):
        y_ndc = 1.0 - 2.0 * (j + 0.5) / height
        row = []
        last_lvl = -1
        for i in range(width):
            x_ndc = 2.0 * (i + 0.5) / width - 1.0

            # 屏幕射线（世界坐标），相机在 +z 方向看向原点
            dx = x_ndc * half_w
            dy = y_ndc * tan_half
            dz = -1.0
            n = math.sqrt(dx * dx + dy * dy + 1.0)
            dx /= n
            dy /= n
            dz /= n

            # 旋转到物体坐标系（等价于物体绕 y 轴转 angle）
            rdx = dx * ca - dz * sa
            rdz = dx * sa + dz * ca
            ox = -dist * sa
            oz = dist * ca

            t = march(ox, 0.0, oz, rdx, dy, rdz)
            if t is None:
                row.append(" ")
                continue

            px = ox + rdx * t
            py = dy * t
            pz = oz + rdz * t
            nx, ny, nz = normal_at(px, py, pz)

            # 法线转回世界坐标
            wnx = nx * ca + nz * sa
            wnz = -nx * sa + nz * ca

            # 交点也转回世界坐标，再算「指向相机」的视线方向
            wx = px * ca + pz * sa
            wz = -px * sa + pz * ca
            vx = -wx
            vy = -py
            vz = dist - wz
            vl = math.sqrt(vx * vx + vy * vy + vz * vz)
            if vl > 0.0:
                vx /= vl
                vy /= vl
                vz /= vl

            val = shade(wnx, ny, wnz, vx, vy, vz)

            idx = int(val * level_count + 0.5)
            if idx > level_count:
                idx = level_count
            if color:
                lvl = int(val * pal_count + 0.5)
                if lvl > pal_count:
                    lvl = pal_count
                # 只在颜色变化时输出转义，减少终端写入量
                if lvl != last_lvl:
                    row.append(escapes[lvl])
                    last_lvl = lvl
                row.append(RAMP[idx])
            else:
                row.append(RAMP[idx])
        if color:
            row.append("\x1b[0m")
        lines.append("".join(row))
    return lines


def enable_ansi():
    """Windows 10 上开启 VT 转义支持。"""
    if os.name == "nt":
        os.system("")


def main():
    parser = argparse.ArgumentParser(
        description="在终端里旋转显示一个 3D 爱心（纯标准库，无依赖）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-W", "--width", type=int, default=0, help="宽度（字符数），0 自动")
    parser.add_argument("-H", "--height", type=int, default=0, help="高度（字符数），0 自动")
    parser.add_argument("--fps", type=float, default=12.0, help="动画帧率上限")
    parser.add_argument("--zoom", type=float, default=1.0, help="缩放倍数")
    parser.add_argument("--static", action="store_true", help="只渲染一帧后退出")
    parser.add_argument("--angle", type=float, default=0.55, help="--static 时的角度（弧度）")
    parser.add_argument("--no-color", action="store_true", help="禁用 ANSI 真彩色")
    args = parser.parse_args()

    color = not args.no_color
    if color:
        enable_ansi()

    term = shutil.get_terminal_size((80, 24))
    width = args.width or min(term.columns - 1, 96)
    height = args.height or min(term.lines - 2, max(18, int(width * 0.46)))
    width = max(width, 12)
    height = max(height, 8)

    if args.static:
        for line in render(width, height, args.angle, color, args.zoom):
            print(line)
        return 0

    sys.stdout.write("\x1b[?25l\x1b[2J")
    sys.stdout.flush()

    angle = 0.0
    delay = 1.0 / max(args.fps, 1.0)
    try:
        while True:
            start = time.time()
            lines = render(width, height, angle, color, args.zoom)
            buf = ["\x1b[H"]
            buf.extend(line + "\n" for line in lines)
            sys.stdout.write("".join(buf))
            sys.stdout.flush()
            angle += 0.07
            if angle > 2.0 * math.pi:
                angle -= 2.0 * math.pi
            rest = delay - (time.time() - start)
            if rest > 0:
                time.sleep(rest)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\x1b[0m\x1b[?25h\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
