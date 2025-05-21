# visualization_utils.py
import cv2
import numpy as np
import config # Importa as configurações

def draw_roi_on_frame(frame, roi_points):
    if roi_points:
        cv2.polylines(frame, [np.array(roi_points, dtype=np.int32)],
                      isClosed=True, color=config.ROI_COLOR, thickness=2)

def draw_detected_balls(frame, detected_balls_info):
    """Desenha as bolas detectadas inicialmente."""
    for ball in detected_balls_info:
        center = ball['center']
        radius = ball['radius']
        cv2.circle(frame, center, radius, config.BALL_DETECT_COLOR, config.BOUNDING_BOX_THICKNESS)
        cv2.circle(frame, center, 2, config.BALL_DETECT_CENTER_COLOR, -1)
        cv2.putText(frame, f"ID: {ball['id']}",
                    (center[0] - radius, center[1] - radius - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.BALL_DETECT_COLOR, 1)

def draw_tracked_objects(frame, tracked_objects_info, draw_trajectory=True):
    """Desenha os objetos rastreados (bounding box, ID, trajetória)."""
    for t_info in tracked_objects_info: # Iterar sobre todos para desenhar trajetórias mesmo se inativos
        if t_info['active']: # Desenha bbox e ID apenas para ativos
            bbox = t_info['bbox']
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, t_info['color'], config.BOUNDING_BOX_THICKNESS, 1)
            cv2.putText(frame, f"ID: {t_info['id']}", (p1[0], p1[1] - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, t_info['color'], 1)

        if draw_trajectory and len(t_info['trajectory']) > 1:
            for i in range(1, len(t_info['trajectory'])):
                if t_info['trajectory'][i - 1] is None or t_info['trajectory'][i] is None:
                    continue
                cv2.line(frame, t_info['trajectory'][i - 1], t_info['trajectory'][i],
                         t_info['color'], config.TRAJECTORY_LINE_THICKNESS)

def draw_hud(frame, fps, frame_count, tracked_count, total_initial_trackers):
    """Desenha informações na tela (Heads-Up Display)."""
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.HUD_TEXT_COLOR, 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.HUD_TEXT_COLOR, 2)
    cv2.putText(frame, f"Trackers: {tracked_count}/{total_initial_trackers}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.HUD_TEXT_COLOR, 2)

def resize_display_frame(frame, window_name):
    """Redimensiona a janela se o frame for maior que WINDOW_MAX_HEIGHT."""
    h, w = frame.shape[:2]
    if h > config.WINDOW_MAX_HEIGHT:
        aspect_ratio = w / h
        display_width = int(config.WINDOW_MAX_HEIGHT * aspect_ratio)
        cv2.resizeWindow(window_name, display_width, config.WINDOW_MAX_HEIGHT)