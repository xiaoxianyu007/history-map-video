"""
战国七雄地图数据 + 渲染引擎 v2
使用 cartopy + Natural Earth 真实地理数据
"""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import natural_earth

# ============================================================
# 配色方案（古风）
# ============================================================
COLORS = {
    "qin":   "#C41E3A",   # 秦 朱红
    "han":   "#D4952A",   # 韩 琥珀
    "zhao":  "#2D7D46",   # 赵 黛绿
    "wei":   "#3A6B9E",   # 魏 靛蓝
    "chu":   "#7B3D8C",   # 楚 绛紫
    "yan":   "#3D5C5C",   # 燕 青灰
    "qi":    "#B8860B",   # 齐 暗金
    "bg":    "#F2E8D0",   # 背景 古纸色
    "sea":   "#C8DCE8",   # 海洋 淡蓝
    "land":  "#E8DCC8",   # 陆地 浅土
    "conquered": "#9A9A8A",  # 被灭国 枯灰
    "border_line": "#8B7355",  # 疆界线 棕
}

# 七国名称
STATE_NAMES = {
    "qin":  "秦",  "han":  "韩",  "zhao": "赵",
    "wei":  "魏",  "chu":  "楚",  "yan":  "燕",  "qi":   "齐",
}

# 都城坐标
CAPITALS = {
    "qin":  (108.9, 34.3),   # 咸阳
    "han":  (113.1, 35.1),   # 新郑
    "zhao": (114.5, 38.0),   # 邯郸
    "wei":  (114.3, 36.7),   # 大梁
    "chu":  (112.2, 30.4),   # 郢都
    "yan":  (116.4, 39.9),   # 蓟
    "qi":   (118.0, 36.9),   # 临淄
}

CONQUEST_ORDER = ["han", "zhao", "wei", "chu", "yan", "qi"]

# ============================================================
# 疆域多边形（更细致的边界，带自然弯曲）
# ============================================================
TERRITORIES = {
    "qin": [
        (103.2, 36.8), (103.8, 37.8), (104.8, 38.2), (105.8, 38.5),
        (106.8, 38.2), (108.0, 38.0), (109.2, 37.8), (109.8, 37.2),
        (110.5, 36.5), (110.8, 35.8), (111.0, 35.0), (110.5, 34.2),
        (109.8, 33.8), (109.0, 33.5), (108.2, 33.2), (107.2, 33.0),
        (106.2, 33.2), (105.2, 33.8), (104.5, 34.2), (104.0, 34.8),
        (103.5, 35.5), (103.0, 36.2),
    ],
    "han": [
        (111.0, 35.8), (111.8, 36.0), (112.8, 36.0), (113.5, 35.6),
        (113.8, 35.0), (113.5, 34.5), (113.0, 34.2), (112.2, 34.0),
        (111.5, 34.0), (111.0, 34.2), (110.5, 34.5), (110.5, 35.2),
    ],
    "zhao": [
        (110.5, 39.5), (111.5, 40.5), (112.8, 40.8), (114.0, 41.0),
        (115.5, 41.0), (117.0, 40.8), (118.5, 40.5), (119.5, 39.8),
        (120.0, 39.2), (119.8, 38.5), (119.0, 37.8), (118.2, 37.2),
        (117.5, 37.0), (116.8, 36.8), (115.8, 36.5), (114.5, 36.5),
        (113.5, 36.5), (112.5, 36.8), (111.8, 37.2), (111.2, 37.8),
        (110.8, 38.5), (110.5, 39.0),
    ],
    "wei": [
        (111.0, 36.8), (112.0, 37.0), (113.0, 37.0), (114.2, 37.0),
        (115.2, 36.8), (115.8, 36.5), (116.2, 36.0), (116.0, 35.5),
        (115.5, 35.2), (114.8, 35.0), (114.0, 35.0), (113.0, 35.0),
        (112.0, 35.0), (111.5, 35.2), (111.0, 35.5), (110.8, 36.0),
    ],
    "chu": [
        (107.0, 32.8), (108.5, 33.2), (109.8, 33.5), (111.0, 33.5),
        (112.0, 33.2), (113.5, 33.0), (115.0, 32.8), (116.5, 32.2),
        (117.8, 31.5), (119.0, 30.8), (120.0, 30.0), (120.2, 29.2),
        (119.5, 28.2), (118.5, 27.5), (117.0, 27.2), (115.5, 27.0),
        (114.0, 27.0), (112.5, 27.2), (111.0, 27.5), (109.8, 28.0),
        (108.8, 28.8), (108.0, 29.5), (107.2, 30.5), (106.8, 31.5),
        (106.8, 32.2),
    ],
    "yan": [
        (115.0, 41.5), (116.0, 42.0), (117.5, 42.0), (119.0, 42.0),
        (120.5, 41.8), (122.0, 41.5), (123.5, 41.2), (124.5, 40.8),
        (124.8, 40.0), (124.2, 39.5), (123.0, 39.2), (121.5, 39.0),
        (120.0, 39.0), (118.8, 39.2), (117.5, 39.5), (116.5, 40.0),
        (115.5, 40.5),
    ],
    "qi": [
        (115.0, 37.8), (116.0, 38.2), (117.5, 38.5), (119.0, 38.5),
        (120.5, 38.2), (121.5, 37.8), (122.5, 37.2), (122.8, 36.5),
        (122.5, 35.8), (121.5, 35.2), (120.5, 35.0), (119.5, 35.0),
        (118.5, 35.2), (117.5, 35.5), (116.8, 36.0), (116.0, 36.5),
        (115.5, 37.0),
    ],
}

# 地图视口（包含整个战国区域）
MAP_EXTENT = (102, 126, 26, 43)

# ============================================================
# 中文字体
# ============================================================
_CN_FONTS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
]


def _get_cn_font(size=20):
    for p in _CN_FONTS:
        try:
            return FontProperties(fname=p, size=size)
        except Exception:
            continue
    return FontProperties(family="sans-serif", size=size)


# ============================================================
# 生成地形底图（使用噪声模拟）
# ============================================================
def _make_terrain(extent, shape=(400, 300)):
    """生成一个类似地形的底图，使用分形噪声"""
    import noise  # 可选
    lon_min, lon_max, lat_min, lat_max = extent
    lons = np.linspace(lon_min, lon_max, shape[0])
    lats = np.linspace(lat_min, lat_max, shape[1])
    X, Y = np.meshgrid(lons, lats)

    try:
        # 使用噪声库产生真实感地形
        scale = 3.0
        Z = np.zeros_like(X)
        for i in range(shape[1]):
            for j in range(shape[0]):
                Z[i, j] = noise.pnoise2(
                    X[0, j] * scale, Y[i, 0] * scale,
                    octaves=6, persistence=0.5, lacunarity=2.0,
                )
    except ImportError:
        # fallback: 使用叠加正弦波
        Z = (
            0.5 * np.sin(X * 1.5) * np.cos(Y * 1.8)
            + 0.3 * np.sin(X * 3.0 + Y * 2.5)
            + 0.2 * np.cos(X * 5.0) * np.sin(Y * 4.0)
            + 0.1 * np.sin(X * 8.0 + Y * 7.0)
        )
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    return X, Y, Z


# ============================================================
# 核心渲染函数 v2 - 使用 cartopy
# ============================================================
def render_frame(
    output_path: str,
    conquered: set[str],
    target: str | None,
    progress: float = 0.0,
    seg_index: int = 0,
    dpi: int = 100,
    figsize: tuple = (16, 9),
):
    """渲染一帧地图，使用 cartopy 真实地理数据"""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")

    # 创建地图 - 使用 PlateCarree 投影
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor=COLORS["bg"])
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    lon_min, lon_max, lat_min, lat_max = MAP_EXTENT
    ax.set_extent(MAP_EXTENT, crs=proj)

    # ---- 海洋背景 ----
    ax.add_feature(cfeature.OCEAN, facecolor=COLORS["sea"], zorder=0)
    ax.add_feature(cfeature.LAND, facecolor=COLORS["land"], alpha=0.4, zorder=1)

    # ---- 地形效果 ----
    try:
        X, Y, Z = _make_terrain(MAP_EXTENT, shape=(300, 200))
        cmap_terrain = LinearSegmentedColormap.from_list(
            "terrain",
            [(0.55, 0.45, 0.30, 0.0),   # 低: 透明
             (0.50, 0.40, 0.25, 0.15),  # 中: 浅棕
             (0.40, 0.30, 0.15, 0.30),  # 高: 深棕
             (0.30, 0.20, 0.10, 0.35)], # 最高
        )
        ax.contourf(X, Y, Z, levels=12, cmap=cmap_terrain,
                     transform=proj, zorder=2, alpha=0.6,
                     antialiased=True)
    except Exception:
        pass

    # ---- 真实海岸线 ----
    ax.add_feature(
        cfeature.NaturalEarthFeature("physical", "coastline", "50m",
                                      facecolor="none",
                                      edgecolor="#6B8E9E", linewidth=1.2),
        zorder=3,
    )

    # ---- 真实河流 ----
    rivers = cfeature.NaturalEarthFeature(
        "physical", "rivers_lake_centerlines", "110m",
        facecolor="none",
        edgecolor="#7BA8C8", linewidth=0.8, alpha=0.6,
    )
    ax.add_feature(rivers, zorder=3)

    # ---- 湖泊 ----
    lakes = cfeature.NaturalEarthFeature(
        "physical", "lakes", "110m",
        facecolor=COLORS["sea"], edgecolor="#6B9AB8", linewidth=0.5,
    )
    ax.add_feature(lakes, zorder=3)

    # ---- 绘制各国领土 ----
    all_states = list(TERRITORIES.keys())

    for sid in all_states:
        if sid == "qin":
            continue  # 秦国后绘（更高zorder）

        poly_pts = np.array(TERRITORIES[sid])

        if sid in conquered:
            color = COLORS["conquered"]
            alpha = 0.50
            edge_color = "#888888"
            linewidth = 1.5
        elif sid == target:
            if progress < 0.4:
                color = COLORS[sid]
                alpha = 0.82
                edge_color = "#FFD700"
                linewidth = 3.0
            elif progress < 0.6:
                t = (progress - 0.4) / 0.2
                c_from = np.array(matplotlib.colors.to_rgb(COLORS[sid]))
                c_to = np.array(matplotlib.colors.to_rgb(COLORS["conquered"]))
                color = c_from * (1 - t) + c_to * t
                alpha = 0.82 - t * 0.32
                edge_color = "#FFD700" if t < 0.5 else "#888888"
                linewidth = 3.0 if t < 0.5 else 1.5
            else:
                color = COLORS["conquered"]
                alpha = 0.50
                edge_color = "#888888"
                linewidth = 1.5
        else:
            color = COLORS[sid]
            alpha = 0.70
            edge_color = "#5C4A32"
            linewidth = 2.0

        poly = MplPolygon(poly_pts, closed=True, facecolor=color,
                          edgecolor=edge_color, linewidth=linewidth,
                          alpha=alpha, zorder=10, transform=proj)
        ax.add_patch(poly)

        # 国家名称
        if sid not in conquered or (sid == target and progress < 0.6):
            cx = np.mean(poly_pts[:, 0])
            cy = np.mean(poly_pts[:, 1])
            font_sz = 28 if sid == target else 22
            weight = "bold" if sid == target else "normal"
            ax.text(cx, cy, STATE_NAMES[sid],
                    fontsize=font_sz, color="white", weight=weight,
                    ha="center", va="center",
                    fontproperties=_get_cn_font(font_sz),
                    transform=proj, zorder=12,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor=(0, 0, 0, 0.35),
                              edgecolor="none"))

    # ---- 绘制秦国（始终在顶层） ----
    qin_pts = np.array(TERRITORIES["qin"])
    qin_poly = MplPolygon(qin_pts, closed=True,
                           facecolor=COLORS["qin"],
                           edgecolor="#8B0000", linewidth=2.5,
                           alpha=0.78, zorder=11, transform=proj)
    ax.add_patch(qin_poly)
    qin_cx, qin_cy = np.mean(qin_pts[:, 0]), np.mean(qin_pts[:, 1])
    ax.text(qin_cx, qin_cy, "秦", fontsize=26, color="white",
            weight="bold", ha="center", va="center",
            fontproperties=_get_cn_font(26),
            transform=proj, zorder=13,
            bbox=dict(boxstyle="round,pad=0.15",
                      facecolor=(0, 0, 0, 0.35), edgecolor="none"))

    # ---- 都城标记 ----
    for sid, (clon, clat) in CAPITALS.items():
        if sid == "qin":
            ax.plot(clon, clat, marker="s", color="#FFD700",
                    markersize=10, markeredgecolor="#8B4513",
                    markeredgewidth=1.5, transform=proj, zorder=15)
            ax.text(clon, clat - 0.7, "咸阳",
                    fontsize=11, color="#FFD700", weight="bold",
                    ha="center", fontproperties=_get_cn_font(11),
                    transform=proj, zorder=15)
        else:
            conquered_suffix = sid in conquered or (sid == target and progress >= 0.6)
            if conquered_suffix:
                ax.plot(clon, clat, marker="o", color="#AAAAAA",
                        markersize=5, markeredgecolor="#888888",
                        markeredgewidth=1, transform=proj, zorder=14)
            else:
                ax.plot(clon, clat, marker="o", color="#FFD700",
                        markersize=7, markeredgecolor="#8B4513",
                        markeredgewidth=1, transform=proj, zorder=14)

    # ---- 年份 ----
    years = ["", "公元前230年", "公元前228年", "公元前225年",
             "公元前224年", "公元前222年", "公元前221年", "公元前221年"]
    year_text = years[seg_index] if seg_index < len(years) else ""
    if year_text:
        ax.text(0.98, 0.97, year_text, fontsize=34, color="#4A3728",
                ha="right", va="top", weight="bold",
                transform=ax.transAxes, zorder=20,
                fontproperties=_get_cn_font(34))

    # ---- 标题 ----
    if seg_index == 0:
        ax.text(0.5, 0.96, "秦灭六国之战", fontsize=40, color="#4A3728",
                ha="center", va="top", weight="bold",
                transform=ax.transAxes, zorder=20,
                fontproperties=_get_cn_font(40))

    # ---- 图例信息（左下角） ----
    legend_states = []
    if seg_index == 0:
        legend_states = ["秦", "韩", "赵", "魏"]
    elif seg_index == 1:
        legend_states = ["秦", "赵", "魏", "楚"]
    elif seg_index == 2:
        legend_states = ["秦", "魏", "楚", "燕"]
    elif seg_index == 3:
        legend_states = ["秦", "楚", "燕", "齐"]
    elif seg_index == 4:
        legend_states = ["秦", "燕", "齐"]
    elif seg_index == 5:
        legend_states = ["秦", "齐"]
    elif seg_index in (6, 7):
        legend_states = ["秦"]

    if legend_states:
        legend_str = " · ".join(legend_states)
        ax.text(0.02, 0.03, f"尚未臣服: {legend_str}", fontsize=12,
                color="#5A4A38", ha="left", va="bottom",
                transform=ax.transAxes, zorder=20,
                fontproperties=_get_cn_font(12))

    conquest_count = len(conquered)
    if conquest_count > 0:
        ax.text(0.98, 0.03, f"已灭 {conquest_count} 国", fontsize=12,
                color="#8B0000", ha="right", va="bottom",
                transform=ax.transAxes, zorder=20,
                fontproperties=_get_cn_font(12))

    # ---- 坐标轴隐藏 ----
    ax.axis("off")

    # ---- 添加边框效果 ----
    ax.spines["geo"].set_visible(True)
    ax.spines["geo"].set_edgecolor("#8B7355")
    ax.spines["geo"].set_linewidth(2)

    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=COLORS["bg"], edgecolor="none",
                pad_inches=0.1)
    plt.close(fig)
