import cv2
import dlib
import numpy as np
import sys

# Facial landmark model
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def get_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        raise ValueError("No face detected")
    shape = predictor(gray, faces[0])
    return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.int32)

def rect_contains(rect, point):
    return rect[0] <= point[0] <= rect[0] + rect[2] and rect[1] <= point[1] <= rect[1] + rect[3]

def delaunay_triangulation(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert((float(p[0]), float(p[1])))
    triangle_list = subdiv.getTriangleList()
    triangles = []
    for t in triangle_list:
        pts = [(int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (int(t[4]), int(t[5]))]
        idx = []
        for pt in pts:
            for i, point in enumerate(points):
                if np.linalg.norm(np.array(pt) - point) < 1.0:
                    idx.append(i)
        if len(idx) == 3:
            triangles.append(tuple(idx))
    return triangles

def warp_triangle(img1, img2, t1, t2):
    t1 = np.array(t1, dtype=np.float32).reshape(-1, 2)
    t2 = np.array(t2, dtype=np.float32).reshape(-1, 2)
    if t1.shape[0] != 3 or t2.shape[0] != 3:
        return

    r1 = cv2.boundingRect(t1)
    r2 = cv2.boundingRect(t2)

    t1_rect = []
    t2_rect = []
    t2_rect_int = []
    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2_rect_int.append((int(t2[i][0] - r2[0]), int(t2[i][1] - r2[1])))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0), 16, 0)

    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    if img1_rect.shape[0] == 0 or img1_rect.shape[1] == 0:
        return

    size = (r2[2], r2[3])
    mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    warped = cv2.warpAffine(img1_rect, mat, size, None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    warped = warped * mask

    img2_rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    img2_rect = img2_rect * (1.0 - mask)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2_rect + warped

def apply_seamless_clone(src_face, dst_img, dst_points):
    mask = np.zeros(dst_img.shape, dtype=dst_img.dtype)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.array(dst_points)), (255, 255, 255))
    center = tuple(np.mean(dst_points, axis=0).astype(int))
    return cv2.seamlessClone(src_face, dst_img, mask, center, cv2.NORMAL_CLONE)

def swap_faces(src_img, dst_img, src_points, dst_points, triangles):
    dst_img_copy = dst_img.copy()
    for tri in triangles:
        t1 = [src_points[tri[0]], src_points[tri[1]], src_points[tri[2]]]
        t2 = [dst_points[tri[0]], dst_points[tri[1]], dst_points[tri[2]]]
        warp_triangle(src_img, dst_img_copy, t1, t2)
    result = apply_seamless_clone(dst_img_copy, dst_img, dst_points)
    return result

def main(src_path, dst_path, output_path):
    src_img = cv2.imread(src_path)
    dst_img = cv2.imread(dst_path)
    src_points = get_landmarks(src_img)
    dst_points = get_landmarks(dst_img)
    h, w = dst_img.shape[:2]
    rect = (0, 0, w, h)
    triangles = delaunay_triangulation(rect, dst_points)
    output = swap_faces(src_img, dst_img, src_points, dst_points, triangles)
    cv2.imwrite(output_path, output)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 face_swapper.py source.jpg target.jpg output.jpg")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
