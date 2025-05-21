# ball_detector.py
import cv2
import numpy as np
import config # Importa as configurações

def detect_balls_hough(frame, roi_polygon_points, roi_mask, next_ball_id_start=0):
    """
    Detecta bolas usando HoughCircles dentro da ROI especificada.
    Retorna uma lista de dicionários, cada um representando uma bola detectada.
    """
    if roi_mask is None or roi_polygon_points is None:
        print("ERRO [Detector]: ROI mask ou pontos não fornecidos.")
        return [], next_ball_id_start

    frame_in_roi = cv2.bitwise_and(frame, frame, mask=roi_mask)
    gray = cv2.cvtColor(frame_in_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, config.HOUGH_GRAY_BLUR_KERNEL, config.HOUGH_GRAY_BLUR_SIGMA_X)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=config.HOUGH_DP,
        minDist=config.HOUGH_MIN_DIST,
        param1=config.HOUGH_PARAM1,
        param2=config.HOUGH_PARAM2,
        minRadius=config.HOUGH_MIN_RADIUS,
        maxRadius=config.HOUGH_MAX_RADIUS
    )

    detected_balls_info = []
    current_id = next_ball_id_start

    if circles is not None:
        circles = np.uint16(np.around(circles))
        roi_polygon_np = np.array(roi_polygon_points, dtype=np.int32)
        for c_data in circles[0, :]:
            center_x, center_y, radius = int(c_data[0]), int(c_data[1]), int(c_data[2])

            # Verifica se o centro do círculo está dentro da ROI poligonal
            if cv2.pointPolygonTest(roi_polygon_np, (center_x, center_y), False) >= 0:
                ball_info = {
                    'id': current_id,
                    'center': (center_x, center_y),
                    'radius': radius,
                    'bbox_initial': (center_x - radius, center_y - radius, 2 * radius, 2 * radius)
                }
                detected_balls_info.append(ball_info)
                current_id += 1
    else:
        print("[Detector] Nenhum círculo encontrado por HoughCircles.")

    print(f"[Detector] Bolas detectadas e validadas na ROI: {len(detected_balls_info)}")
    return detected_balls_info, current_id