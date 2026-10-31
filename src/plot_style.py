from pathlib import Path
import os


def configure_plot_style():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    for filename in ('msyh.ttc', 'simhei.ttf'):
        path = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts' / filename
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']
    matplotlib.rcParams['font.sans-serif'] = [name for name in preferred if name in available] + ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['pdf.fonttype'] = 42
