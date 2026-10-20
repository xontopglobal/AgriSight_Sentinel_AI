
from app.vision import analyze_leaf_image

result = analyze_leaf_image("sample_leaf.png")

for key, value in result.items():
    print(f"{key}: {value}")
