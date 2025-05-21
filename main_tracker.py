# main_tracker.py
import cv2
import time
import numpy as np # Necessário para np.array em algumas chamadas de visualização
from ball_tracker_project.config import *
from ball_tracker_project.roi_handler import ROIHandler
from ball_tracker_project.ball_detector import detect_balls_hough
from ball_tracker_project.tracker_manager import TrackerManager
import ball_tracker_project.visualization_utils as viz

def preprocess_frame(frame):
    """Aplica pré-processamentos consistentes ao frame (ex: rotação)."""
    if Config.ROTATE_VIDEO_CLOCKWISE:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame

def main():
    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir o vídeo em '{Config.VIDEO_PATH}'")
        return

    ret, first_frame_raw = cap.read()
    if not ret:
        print("Erro: Não foi possível ler o primeiro frame.")
        cap.release()
        return
    
    first_frame = preprocess_frame(first_frame_raw)

    # 1. Gerenciamento da ROI
    roi_manager = ROIHandler(first_frame)
    if not roi_manager.load_roi():
        print("Nenhuma ROI salva encontrada ou falha ao carregar. Iniciando seleção manual.")
        if not roi_manager.select_roi_interactively():
            print("ROI não definida. Encerrando.")
            cap.release()
            return
        roi_manager.save_roi()
    else:
        reselect = input("ROI carregada. Deseja definir uma nova? (s/n, padrão n): ").lower()
        if reselect == 's':
            if not roi_manager.select_roi_interactively():
                print("ROI não definida após tentativa de reseleção. Encerrando.")
                cap.release()
                return
            roi_manager.save_roi()

    roi_points = roi_manager.get_points()
    roi_mask = roi_manager.get_mask(first_frame.shape)

    if roi_points is None or roi_mask is None:
        print("Falha ao obter ROI. Encerrando.")
        cap.release()
        return

    # Visualização da ROI no primeiro frame
    # frame_with_roi_viz = first_frame.copy()
    # viz.draw_roi_on_frame(frame_with_roi_viz, roi_points)
    # cv2.imshow("ROI Selecionada", frame_with_roi_viz)
    # print("Pressione qualquer tecla para iniciar a detecção de bolas...")
    # cv2.waitKey(0)
    # cv2.destroyWindow("ROI Selecionada")


    # 2. Detecção Inicial de Bolas
    print("\n--- Detecção Inicial de Bolas ---")
    initial_balls, next_id = detect_balls_hough(first_frame, roi_points, roi_mask)
    if not initial_balls:
        print("Nenhuma bola detectada inicialmente. Encerrando.")
        cap.release()
        return

    # Visualização da detecção inicial
    frame_with_initial_detections = first_frame.copy()
    viz.draw_roi_on_frame(frame_with_initial_detections, roi_points)
    viz.draw_detected_balls(frame_with_initial_detections, initial_balls)
    cv2.namedWindow("Detecção Inicial", cv2.WINDOW_NORMAL)
    viz.resize_display_frame(frame_with_initial_detections, "Detecção Inicial")
    cv2.imshow("Detecção Inicial", frame_with_initial_detections)
    print("Pressione qualquer tecla para iniciar o tracking...")
    cv2.waitKey(0)
    cv2.destroyWindow("Detecção Inicial")


    # 3. Inicialização dos Trackers
    print("\n--- Inicialização dos Trackers ---")
    tracker_mgr = TrackerManager(tracker_type=Config.TRACKER_TYPE)
    if not tracker_mgr.initialize_trackers(first_frame, initial_balls):
        print("Falha ao inicializar trackers. Encerrando.")
        cap.release()
        return
    
    total_initial_trackers = len(tracker_mgr.get_all_objects_info())

    # 4. Loop Principal de Tracking
    print("\n--- Iniciando Tracking do Vídeo ---")
    frame_count = 0
    output_window_name = "Rastreamento de Bolas"
    cv2.namedWindow(output_window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame_raw = cap.read()
        if not ret:
            print("Fim do vídeo ou erro ao ler frame.")
            break

        current_frame = preprocess_frame(frame_raw)
        frame_count += 1
        start_time = time.time()

        # Atualiza trackers
        successful_updates = tracker_mgr.update_trackers(current_frame)
        
        # Prepara frame para exibição
        display_frame = current_frame.copy()
        viz.draw_roi_on_frame(display_frame, roi_points)
        
        # Obtém informações dos objetos rastreados (ativos e inativos para desenhar trajetórias)
        all_tracked_objects = tracker_mgr.get_all_objects_info()
        viz.draw_tracked_objects(display_frame, all_tracked_objects)

        # Calcula FPS
        fps = 1.0 / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
        
        # Desenha HUD
        viz.draw_hud(display_frame, fps, frame_count, successful_updates, total_initial_trackers)
        
        viz.resize_display_frame(display_frame, output_window_name)
        cv2.imshow(output_window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Loop de tracking interrompido pelo usuário.")
            break
            
    print(f"Processamento concluído. Total de frames processados: {frame_count}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()