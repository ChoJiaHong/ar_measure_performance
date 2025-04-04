from PIL import Image

# 可選的解析度（你也可以自行增加）
resolutions = {
    "1": (1280, 720),
    "2": (1366, 768),
    "3": (1440, 810),
    "4": (1600, 900),
    "5": (1920, 1080)
}

print("請選擇圖片要轉換成的解析度：")
for key, (w, h) in resolutions.items():
    print(f"{key}. {w}x{h}")

choice = input("輸入選項編號（例如 1）: ").strip()
if choice not in resolutions:
    print("無效選項，請重新執行程式。")
    exit()

target_width, target_height = resolutions[choice]

# 載入原始圖片
image_path = "1280hand.jpg"
output_path = f"{target_width}x{target_height}_hand.jpg"

# 開啟圖片並轉換大小
with Image.open(image_path) as img:
    resized_img = img.resize((target_width, target_height), Image.BILINEAR)
    resized_img.save(output_path)

print(f"圖片已轉換並儲存為 {output_path}")
