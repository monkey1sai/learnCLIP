from collections import Counter
import math

# 假設指稱集合
images = {
    "desc1": {"img1", "img2", "img3"},
    "desc2": {"img2", "img3", "img4"},
    "desc3": {"img5", "img6"}
}

# 計算PMI
def compute_pmi(desc1, desc2, images):
    joint_images = images[desc1] & images[desc2]
    prob_joint = len(joint_images) / sum(len(v) for v in images.values())
    prob_desc1 = len(images[desc1]) / sum(len(v) for v in images.values())
    prob_desc2 = len(images[desc2]) / sum(len(v) for v in images.values())
    
    if prob_joint > 0:
        return math.log(prob_joint / (prob_desc1 * prob_desc2))
    else:
        return 0

# 計算指稱相似度
pmi_value = compute_pmi("desc1", "desc2", images)
print(f"PMI(desc1, desc2): {pmi_value}")
