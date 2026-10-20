"""COCO class names in the 91-entry layout used by RF-DETR (gaps are 'N/A')."""

COCO_CLASSES_91 = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella",
    "N/A", "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "N/A", "dining table", "N/A", "N/A",
    "toilet", "N/A", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "N/A", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# 中文名（用于喂给端侧小模型的提示词，避免模型自己乱翻译）
COCO_CLASSES_ZH = {
    "person": "人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车",
    "bus": "公交车", "truck": "卡车", "train": "火车", "boat": "船",
    "traffic light": "红绿灯", "fire hydrant": "消防栓", "stop sign": "停止标志",
    "bench": "长椅", "cat": "猫", "dog": "狗", "chair": "椅子", "couch": "沙发",
    "potted plant": "盆栽", "dining table": "桌子", "tv": "电视", "laptop": "笔记本电脑",
    "backpack": "背包", "umbrella": "雨伞", "handbag": "手提包", "suitcase": "行李箱",
    "bottle": "瓶子", "cup": "杯子", "book": "书", "cell phone": "手机",
}


def class_name(idx: int) -> str:
    if 0 <= idx < len(COCO_CLASSES_91):
        return COCO_CLASSES_91[idx]
    return f"class_{idx}"


def class_name_zh(name: str) -> str:
    return COCO_CLASSES_ZH.get(name, name)
