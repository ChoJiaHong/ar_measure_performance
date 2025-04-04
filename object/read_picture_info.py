from PIL import Image

# 圖片路徑
image_path = "/home/hiro/git_repo/ar_emulator/1280hand.jpg"

# 開啟圖片
with Image.open(image_path) as img:
    print(f"檔案格式：{img.format}")
    print(f"圖片模式：{img.mode}")
    print(f"圖片尺寸（寬x高）：{img.size}")
    print(f"色彩通道數：{len(img.getbands())}")
