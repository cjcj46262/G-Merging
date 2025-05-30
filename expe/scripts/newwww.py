import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 读取 PDF
pdf_path = "r_zhexian1.pdf"
output_pdf = "r_zhexian11.pdf"

# 创建 PDF 页面
with PdfPages(output_pdf) as pdf:
    fig, ax = plt.subplots(figsize=(8, 6))  # 创建画布
    img = plt.imread(pdf_path)  # 读取 PDF 作为图像（如果 Matplotlib 支持）

    ax.imshow(img)  # 显示图像
    ax.axis("off")  # 关闭坐标轴
    
    pdf.savefig(fig, dpi=300, bbox_inches='tight', pad_inches=0)  # 保存去掉边距的 PDF
    # plt.savefig(fig, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)