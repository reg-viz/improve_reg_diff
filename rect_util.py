import cv2
import numpy as np

def clone(rects):
    return [(r[0], r[1], r[2], r[3]) for r in rects]

# extract bounding recangular from keypoint list
def create_rects_from_points(points):
    rects = []
    for kp in points:
        xx1 = min([int(p.pt[0]) for p in kp])
        xx2 = max([int(p.pt[0]) for p in kp])
        yy1 = min([int(p.pt[1]) for p in kp])
        yy2 = max([int(p.pt[1]) for p in kp])
        if xx1 < xx2:
            x1 = xx1
            x2 = xx2
        else:
            x2 = xx1
            x1 = xx2
        if yy1 < yy2:
            y1 = yy1
            y2 = yy2
        else:
            y2 = yy1
            y1 = yy2
        rects.append((x1, y1, x2, y2))
    return rects

def shift_rects(rects, v):
    return [(x1 + v[0], y1 + v[1], x2 + v[0], y2 + v[1]) for x1, y1, x2, y2 in rects]

# detect to intersect 2 recangulars
def intersect(r1, r2, margin_threshold=0):
    mx1 = max(r1[0], r2[0])
    mx2 = min(r1[2], r2[2])
    my1 = max(r1[1], r2[1])
    my2 = min(r1[3], r2[3])
    result = mx2 - mx1 + margin_threshold > 0 and my2 - my1 + margin_threshold > 0
    connected_rect = (min(r1[0], r2[0]), min(r1[1], r2[1]), max(r1[2], r2[2]), max(r1[3], r2[3]))
    return (result, connected_rect)

def is_valid_rect(rect):
    w, h = (rect[2] - rect[0], rect[3] - rect[1])
    return w >= 4 and h >= 4 and w * h >= 80

def in_boxes(kp, rects, mt=8):
    result = True
    for box in rects:
        in_box = kp.pt[0] > box[0] - mt and kp.pt[0] < box[2] + mt and kp.pt[1] > box[1] - mt and kp.pt[1] < box[3] + mt
        result = result and (not in_box)
    return result

def marge_rects(input_rects, margin_threshold=0):
    rects = [(r[0], r[1], r[2], r[3]) for r in input_rects]
    connected_pairs1 = [[] for i in range(len(rects))]
    for i in range(len(rects)):
        r1 = rects[i]
        for j in range(i + 1, len(rects)):
            r2 = rects[j]
            result, connected = intersect(r1, r2, margin_threshold)
            if result:
                connected_pairs1[i].append(j)
                rects[i] = connected
                r1 = connected
                rects[j] = connected
    return [rects[i] for i in range(len(connected_pairs1)) if len(connected_pairs1[i]) == 0]

def marge_rects_if_same_center(input_rects, centers, margin_threshold=0):
    rects = [(r[0], r[1], r[2], r[3]) for r in input_rects]
    connected_pairs1 = [[] for i in range(len(rects))]
    for i in range(len(rects)):
        r1 = rects[i]
        for j in range(i + 1, len(rects)):
            r2 = rects[j]
            if not centers[i][0] == centers[j][0] or not centers[i][1] == centers[j][1]:
                continue
            result, connected = intersect(r1, r2, margin_threshold)
            if result:
                connected_pairs1[i].append(j)
                rects[i] = connected
                r1 = connected
                rects[j] = connected
    return [(rects[i], centers[i]) for i in range(len(connected_pairs1)) if len(connected_pairs1[i]) == 0]

def volume(rect):
    x1, y1, x2, y2 = rect
    return (x2 - x1) * (y2 - y1)

def filter_intersections(in_rects1, in_rects2, centers):
    size = len(in_rects1)
    marked_index = set([])
    for i in range(size):
        if i in marked_index:
            continue
        r11 = in_rects1[i]
        r21 = in_rects2[i]
        for j in range(i + 1, size):
            if j in marked_index:
                continue
            r12 = in_rects1[j]
            r22 = in_rects2[j]
            if intersect(r11, r12)[0] or intersect(r21, r22)[0]:
                v1 = volume(r11)
                v2 = volume(r12)
                if v1 > v2:
                    mi = j
                else:
                    mi = i
                marked_index.add(mi)
    new_rects1 = [in_rects1[i] for i in range(size) if not i in marked_index]
    new_rects2 = [in_rects2[i] for i in range(size) if not i in marked_index]
    new_centers = [centers[i] for i in range(size) if not i in marked_index]
    return (new_rects1, new_rects2, new_centers)


def allclose(img1, r1, img2, r2, dr=2, sv=None):
    x1, y1, x2, y2 = (r1[0], r1[1], r2[0], r2[1])
    w1, h1 = (r1[2] - r1[0], r1[3] - r1[1])
    w2, h2 = (r2[2] - r2[0], r2[3] - r2[1])
    if not abs(w1 - w2) <= dr or not abs(h1 - h2) <= dr:
        # FIXME
        return (True, None, None, None)
    w, h = (min(w1, w2), min(h1, h2))
    for dx, dy in [sv] if sv else [(sx, sy) for sx in range(-dr, dr+1) for sy in range(-dr, dr+1)]:
        if x2 + dx < 0 or x2 + dx + w >= img2.shape[1] or y1 + dy < 0 or y2 + dy + h >= img2.shape[0] :
            continue
        imgr1 = img1[y1:y1+h, x1:x1+w]
        imgr2 = img2[y2+dy:y2+dy+h, x2+dx:x2+dx+w]
        result = np.allclose(imgr1, imgr2)
        if result:
            return (True, imgr1, imgr2, (dx, dy))
    imgr1 = img1[y1:y1+h, x1:x1+w]
    imgr2 = img2[y2:y2+h, x2:x2+w]
    return (False, imgr1, imgr2, (0, 0))

def nonzero_rects(input, dx, dy, reverse=False):
    h, w = input.shape
    result = []
    for i in range(int(h / dy) + (0 if h % dy == 0 else 1)):
        y1, y2 = (i * dy, (i + 1) * dy)
        y2 = y2 if y2 < h else h - 1
        for j in range(int(w / dx) + (0 if w % dx == 0 else 1)):
            x1, x2 = (j * dx, (j + 1) * dx)
            x2 = x2 if x2 < w else w - 1
            part = input[y1:y2,x1:x2]
            non_zero_y = []
            mx1, mx2 = (w, -1)
            my1, my2 = (h, -1)
            for k in range(y1, y2):
                for l in range(x1, x2):
                    if (not reverse and part[k - y1, l - x1] == 0) or (reverse and not part[k - y1, l - x1] == 0):
                        mx1 = min(mx1, l)
                        mx2 = max(mx2, l)
                        my1 = min(my1, k)
                        my2 = max(my2, k)
            if mx1 >= 0 and my1 >= 0 and mx2 - mx1 >= 0 and my2 - my1 >= 0:
                result.append((mx1, my1, mx2, my2))
    return result

def expand(rects, target_index, shape):
    w = shape[1]
    h = shape[0]
    target = rects[target_index]
    x1, y1, x2, y2 = rects[target_index]
    ex1 = max([r[2] for r in rects if not r == target and intersect((0, y1, x2, y2), r, 0)[0]] + [-1]) + 1
    ex2 = min([r[0] for r in rects if not r == target and intersect((x1, y1, w, y2), r, 0)[0]] + [w]) - 1
    ey1 = max([r[3] for r in rects if not r == target and intersect((x1, 0, x2, y2), r, 0)[0]] + [-1]) + 1
    ey2 = min([r[1] for r in rects if not r == target and intersect((x1, y1, x2, h ), r, 0)[0]] + [h]) - 1
    if ex2 < 0:
        print('debug', rects, target_index, shape)
    return (ex1, ey1, ex2, ey2)

def render_rects(in_img, rects, color, w=1, padding=0):
    out_img = in_img
    for rect in rects:
        x1, y1, x2, y2 = rect
        out_img = cv2.rectangle(out_img, (x1 - padding, y1 - padding), (x2 + padding, y2 + padding), color, w)
    return out_img
