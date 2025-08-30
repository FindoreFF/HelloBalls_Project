import os
import json

# ===== 配置路径（请根据你的目录改）=====
JSON_FOLDER = "G:\ISDN3002\d_dataset/annotations/train"
OUTPUT_FOLDER = "G:\ISDN3002\yolo_dataset\labels-4"
CLASS_MAP = {
    "one": 0,
    "call": 1,
    "ok": 2,
    "fist": 3,
    "palm": 4  
}
# ====================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 批量处理每个 .json 文件
for filename in os.listdir(JSON_FOLDER):
    if not filename.endswith('.json'):
        continue

    file_path = os.path.join(JSON_FOLDER, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for uuid, content in data.items():
        bboxes = content.get("bboxes", [])
        labels = content.get("labels", [])

        if not bboxes or not labels:
            continue

        lines = []
        for bbox, label in zip(bboxes, labels):
            if label not in CLASS_MAP:
                continue
            cls_id = CLASS_MAP[label]
            x_center, y_center, width, height = bbox
            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        if lines:
            txt_path = os.path.join(OUTPUT_FOLDER, f"{uuid}.txt")
            with open(txt_path, 'w') as txt_file:
                txt_file.write('\n'.join(lines))

print("✅ 所有 JSON 文件已成功转换为 YOLO 格式！")
