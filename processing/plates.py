# plates.py – phiên bản FIX chuẩn nhất 2025
import cv2
import numpy as np

MIN_AR = 2.0
MAX_AR = 8.5
MIN_AREA = 350
MAX_AREA = 200000

def preprocess(gray):
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.equalizeHist(gray)
    return gray

def deskew_plate(plate_img):
    """Xoay thẳng biển rất quan trọng"""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    pts = cv2.findNonZero(th)
    if pts is None:
        return plate_img

    rect = cv2.minAreaRect(pts)
    angle = rect[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = plate_img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    rot = cv2.warpAffine(plate_img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rot

def crop_map(rect, scale, W0, H0):
    x,y,w,h = rect
    x0 = int(x/scale)
    y0 = int(y/scale)
    w0 = int(w/scale)
    h0 = int(h/scale)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W0, x0+w0)
    y1 = min(H0, y0+h0)
    if x1<=x0 or y1<=y0:
        return None
    return (x0,y0,x1-x0,y1-y0)

def find_and_extract_plate(path):
    img = cv2.imread(path)
    if img is None:
        return None, None

    H0, W0 = img.shape[:2]
    target_w = 900
    scale = target_w / W0
    img_r = cv2.resize(img, (target_w, int(H0*scale)))

    gray = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    gray = preprocess(gray)

    try:
        mser = cv2.MSER_create(_min_area=40, _max_area=7000)
    except:
        mser = cv2.MSER_create()

    regs, _ = mser.detectRegions(gray)
    mask = np.zeros_like(gray)

    for pts in regs:
        hull = cv2.convexHull(pts.reshape(-1,1,2))
        cv2.drawContours(mask, [hull], -1, 255, -1)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30,8))
    dil = cv2.dilate(mask, kernel, 2)

    cnts,_ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best=None; best_sc=0
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ar = w/h
        area = w*h
        if area<MIN_AREA or area>MAX_AREA: continue
        if ar<MIN_AR or ar>MAX_AR: continue
        score = area
        if score>best_sc:
            best_sc = score
            best = (x,y,w,h)

    if best is None:
        return None, None

    mapped = crop_map(best, scale, W0, H0)
    if mapped is None:
        return None, None

    x0,y0,w0,h0 = mapped
    crop = img[y0:y0+h0, x0:x0+w0]

    # 🔥 FIX: bỏ viền trắng và xoay biển
    crop = cv2.copyMakeBorder(crop,10,10,10,10,cv2.BORDER_REPLICATE)
    crop = deskew_plate(crop)

    return crop, (x0,y0,w0,h0)
