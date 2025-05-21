import cv2
import numpy as np
import json
import copy
import os
import time # Para calcular FPS (opcional)

# --- Constantes ---
ROI_CONFIG_FILE = "roi_config.json"

# --- Variáveis Globais ---
roi_points_global = []
current_frame_copy_for_roi = None
original_first_frame_processed = None
roi_defined_global = False
mask_roi_global = None
active_trackers_list = [] # Renomeado para clareza
next_ball_id_counter = 0

# --- Funções para Salvar e Carregar ROI (sem alterações) ---
def save_roi_points(points, filepath=ROI_CONFIG_FILE):
    try:
        with open(filepath, 'w') as f:
            json.dump(points, f)
        print(f"Pontos da ROI salvos em '{filepath}'")
    except Exception as e:
        print(f"Erro ao salvar pontos da ROI: {e}")

def load_roi_points(filepath=ROI_CONFIG_FILE):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                points = json.load(f)
            if isinstance(points, list) and all(isinstance(p, list) and len(p) == 2 for p in points):
                points = [tuple(p) for p in points]
                print(f"Pontos da ROI carregados de '{filepath}': {points}")
                return points
            else:
                print(f"Arquivo '{filepath}' não contém uma lista de pontos válida.")
                return None
        except Exception as e:
            print(f"Erro ao carregar pontos da ROI de '{filepath}': {e}")
            return None
    return None

# --- Função de Callback do Mouse (sem alterações) ---
def select_roi_points_callback(event, x, y, flags, param):
    global roi_points_global, current_frame_copy_for_roi
    if roi_defined_global:
        return
    window_name = param['window_name']
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_points_global.append((x, y))
        cv2.circle(current_frame_copy_for_roi, (x, y), 5, (0, 255, 0), -1)
        if len(roi_points_global) > 1:
            cv2.line(current_frame_copy_for_roi, roi_points_global[-2], roi_points_global[-1], (0, 255, 0), 2)
        cv2.imshow(window_name, current_frame_copy_for_roi)

# --- Função de Detecção de Bolas (sem alterações) ---
def detect_initial_balls(frame_to_process, roi_polygon_points_list, roi_mask_arg):
    global next_ball_id_counter
    print("\n--- Iniciando Detecção Inicial de Bolas ---")
    frame_in_roi = cv2.bitwise_and(frame_to_process, frame_to_process, mask=roi_mask_arg)
    gray = cv2.cvtColor(frame_in_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15, 15), 0.7)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=2.9, minDist=30,
        param1=25, param2=40, minRadius=15, maxRadius=20
    )
    detected_balls_for_tracking = []
    output_frame = frame_to_process.copy()
    roi_polygon_np_array = np.array(roi_polygon_points_list, dtype=np.int32)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c_data in circles[0, :]:
            center_x, center_y, radius = c_data[0], c_data[1], c_data[2]
            if cv2.pointPolygonTest(roi_polygon_np_array, (center_x, center_y), False) >= 0:
                ball_info = {
                    'center': (center_x, center_y), 'radius': radius, 'id': next_ball_id_counter,
                    'bbox_initial': (center_x - radius, center_y - radius, 2 * radius, 2 * radius)
                }
                detected_balls_for_tracking.append(ball_info)
                next_ball_id_counter += 1
                cv2.circle(output_frame, (center_x, center_y), radius, (0, 255, 0), 2)
                cv2.circle(output_frame, (center_x, center_y), 2, (0, 0, 255), 3)
                cv2.putText(output_frame, f"ID: {ball_info['id']}", (center_x - radius, center_y - radius - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    print(f"Total de bolas detectadas para tracking: {len(detected_balls_for_tracking)}")
    cv2.polylines(output_frame, [roi_polygon_np_array], True, (255, 0, 0), 2)
    cv2.namedWindow("Bolas Detectadas (Inicial)", cv2.WINDOW_NORMAL)
    cv2.imshow("Bolas Detectadas (Inicial)", output_frame)
    h_out, w_out = output_frame.shape[:2]; display_height_out = 700
    if h_out > display_height_out:
        cv2.resizeWindow("Bolas Detectadas (Inicial)", int(display_height_out * (w_out/h_out)), display_height_out)
    print("Pressione tecla na janela 'Bolas Detectadas' para trackers.")
    cv2.waitKey(0)
    cv2.destroyWindow("Bolas Detectadas (Inicial)")
    return detected_balls_for_tracking, output_frame

# --- Função de Inicialização dos Trackers (sem alterações) ---
def initialize_trackers(frame_for_init, balls_to_track_info):
    global active_trackers_list
    active_trackers_list = []
    print("\n--- Iniciando Inicialização dos Trackers ---")
    if not balls_to_track_info: return False
    for ball_info in balls_to_track_info:
        try: tracker = cv2.TrackerCSRT_create()
        except AttributeError:
            try: tracker = cv2.legacy.TrackerMOSSE_create()
            except AttributeError:
                print("ERRO: Nenhum tracker (CSRT, MOSSE) encontrado. Verifique OpenCV Contrib."); return False
        x, y, w, h = ball_info['bbox_initial']
        frame_h, frame_w = frame_for_init.shape[:2]
        print(f"Tracker ID {ball_info['id']} - BBox inicial: {x}, {y}, {w}, {h}")
        print(f"Frame: {frame_w}x{frame_h}")
        x, y = max(0, x), max(0, y)
        w, h = min(w, frame_w - x), min(h, frame_h - y)
        print(f"Tracker ID {ball_info['id']} - BBox inicial: {x}, {y}, {w}, {h}")
        if w <= 0 or h <= 0: print(f"AVISO: BBox inválida ID {ball_info['id']}. Pulando."); continue
        adjusted_bbox = (x, y, w, h)
        print(f"Tracker ID {ball_info['id']} - BBox ajustada: {adjusted_bbox}")
        try:
            tracker.init(frame_for_init, adjusted_bbox)  # NÃO atribuir retorno, pois é None
            print(f"Tracker ID {ball_info['id']} inicializado com sucesso.")
            active_trackers_list.append({
                'id': ball_info['id'], 'tracker_obj': tracker, 'bbox': adjusted_bbox,
                'trajectory': [(int(x + w/2), int(y + h/2))],
                'color': tuple(np.random.randint(0, 255, 3).tolist()), 'active': True
            })
        except Exception as e:
            print(f"Exceção init tracker ID {ball_info['id']}: {e}")
    if not active_trackers_list: print("Nenhum tracker inicializado."); return False
    print(f"Total de {len(active_trackers_list)} trackers inicializados."); return True

# --- PARTE FINAL: Loop Principal de Tracking ---
def run_video_tracking(cap, roi_poly_points, roi_mask_for_check=None):
    global active_trackers_list
    frame_count = 0
    fps = 0

    # Obter a largura e altura do frame do vídeo para a janela de saída
    # É importante que seja do cap, não do first_frame processado,
    # caso a rotação tenha mudado as dimensões para o first_frame apenas.
    # Mas os frames do cap virão na orientação original do vídeo.
    # Portanto, precisamos processar cada frame (rotacionar) como fizemos com o first_frame.
    
    # Re-ler o primeiro frame para ter certeza das dimensões após rotação
    # ou usar as dimensões de original_first_frame_processed
    if original_first_frame_processed is not None:
        h_display, w_display = original_first_frame_processed.shape[:2]
    else: # Fallback, pode não ser ideal se houve rotação
        w_display = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_display = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_window_name = "Tracking de Bolas"
    cv2.namedWindow(output_window_name, cv2.WINDOW_NORMAL)
    # Redimensionar a janela de saída se for muito grande
    display_height_max = 720
    if h_display > display_height_max:
        aspect_ratio = w_display / h_display
        cv2.resizeWindow(output_window_name, int(display_height_max * aspect_ratio), display_height_max)

    print(f"Janela de saída: '{output_window_name}' ({w_display}x{h_display})")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou erro ao ler frame.")
            break

        frame_count += 1
        start_time = time.time() # Para cálculo de FPS

        # --- Aplicar a mesma transformação do primeiro frame (rotação) ---
        # Precisamos saber se a rotação foi aplicada ao original_first_frame_processed
        # Esta informação não está explicitamente salva, então vamos assumir que se
        # original_first_frame_processed existe, ele é o modelo.
        # Uma forma mais robusta seria salvar a decisão de rotação.
        # Por agora, vamos comparar as dimensões para inferir se foi rotacionado.
        # Ou, mais simples, apenas replicar a pergunta ou ter uma flag.
        # Assumindo que `original_first_frame_processed` tem a orientação correta:
        if original_first_frame_processed is not None and \
           frame.shape[:2] != original_first_frame_processed.shape[:2] and \
           frame.shape[:2][::-1] == original_first_frame_processed.shape[:2]: # Verifica se transposto
            # Esta lógica de rotação precisa ser consistente com a do first_frame
            # Se o usuário escolheu rotacionar, fazemos aqui também.
            # Para simplificar, vou assumir que se original_first_frame_processed existe,
            # e a rotação o tornou diferente do frame cru, então rotacionamos.
            # Uma flag `was_rotated` seria melhor.
            # Hardcoding a rotação por enquanto se foi feita no início:
            if 'rotate_choice_main' in globals() and rotate_choice_main == 's': # Variável global para guardar a escolha
                 frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)


        current_tracking_frame = frame.copy() # Trabalhar em uma cópia
        trackers_updated_successfully = 0

        for tracker_info in active_trackers_list:
            if not tracker_info['active']:
                continue

            tracker_obj = tracker_info['tracker_obj']
            success, bbox = tracker_obj.update(current_tracking_frame)

            if success:
                trackers_updated_successfully += 1
                tracker_info['bbox'] = bbox
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(current_tracking_frame, p1, p2, tracker_info['color'], 2, 1)
                cv2.putText(current_tracking_frame, f"ID: {tracker_info['id']}",
                            (p1[0], p1[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tracker_info['color'], 1)

                # Atualizar e desenhar trajetória
                center_x = int(bbox[0] + bbox[2] / 2)
                center_y = int(bbox[1] + bbox[3] / 2)
                tracker_info['trajectory'].append((center_x, center_y))

                # Verificar se a bola saiu da ROI (opcional)
                # if roi_mask_for_check is not None:
                #     if cv2.pointPolygonTest(np.array(roi_poly_points, dtype=np.int32), (center_x, center_y), False) < 0:
                #         print(f"Bola ID {tracker_info['id']} saiu da ROI. Desativando tracker.")
                #         tracker_info['active'] = False # Desativa o tracker
                #         continue # Pula para o próximo tracker

                for i in range(1, len(tracker_info['trajectory'])):
                    if tracker_info['trajectory'][i-1] is None or tracker_info['trajectory'][i] is None:
                        continue
                    cv2.line(current_tracking_frame, tracker_info['trajectory'][i-1],
                             tracker_info['trajectory'][i], tracker_info['color'], 2)
            else:
                # Marca o tracker como inativo se falhar
                print(f"Tracker para bola ID {tracker_info['id']} falhou.") # Poderia ser menos verboso
                tracker_info['active'] = False
                # Opcionalmente, podemos tentar re-detectar aqui, mas adiciona complexidade.

        # Desenhar a ROI no frame
        cv2.polylines(current_tracking_frame, [np.array(roi_poly_points, dtype=np.int32)], True, (255, 0, 0), 2)

        # Cálculo e exibição de FPS
        end_time = time.time()
        if (end_time - start_time) > 0:
            fps = 1.0 / (end_time - start_time)
        cv2.putText(current_tracking_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(current_tracking_frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(current_tracking_frame, f"Trackers Ativos: {trackers_updated_successfully}/{len(active_trackers_list)}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        cv2.imshow(output_window_name, current_tracking_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Loop de tracking interrompido pelo usuário.")
            break

    print(f"Processamento concluído. Total de frames processados: {frame_count}")


# --- Função Principal ---
def main():
    global roi_points_global, current_frame_copy_for_roi, original_first_frame_processed
    global roi_defined_global, mask_roi_global, active_trackers_list, next_ball_id_counter
    global rotate_choice_main # Para usar no loop de tracking

    video_path = 'IMG_2798.mp4'
    print(f"Carregando vídeo: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"Erro ao abrir vídeo: {video_path}"); return
    ret, first_frame_raw = cap.read()
    if not ret: print("Erro ao ler primeiro frame."); cap.release(); return
    print("Primeiro frame lido.")

    rotate_choice_main = input("O vídeo ('IMG_2798.mp4') precisa ser rotacionado? (s/n, padrão n): ").lower()
    if rotate_choice_main == 's':
        first_frame_raw = cv2.rotate(first_frame_raw, cv2.ROTATE_90_CLOCKWISE)
        print("Frame rotacionado.")
    original_first_frame_processed = copy.copy(first_frame_raw)

    loaded_points = load_roi_points()
    perform_manual_roi_selection = True
    if loaded_points:
        reselect_choice = input("ROI carregada. Definir nova ROI? (s/n, padrão n): ").lower()
        if reselect_choice != 's':
            roi_points_global = loaded_points
            roi_defined_global = True
            perform_manual_roi_selection = False
        else: roi_points_global = []; roi_defined_global = False

    if perform_manual_roi_selection:
        print("\n--- Seleção Manual da ROI ---")
        current_frame_copy_for_roi = copy.copy(original_first_frame_processed)
        roi_points_global = []; roi_defined_global = False
        win_roi = "Selecione ROI - Esq: Ponto. Enter: Finalizar. r: Reset, q: Sair"
        cv2.namedWindow(win_roi, cv2.WINDOW_NORMAL)
        h_roi, w_roi = current_frame_copy_for_roi.shape[:2]; disp_h_roi = 700
        if h_roi > disp_h_roi: cv2.resizeWindow(win_roi, int(disp_h_roi * (w_roi/h_roi)), disp_h_roi)
        cv2.setMouseCallback(win_roi, select_roi_points_callback, {'window_name': win_roi})
        print(f"- Janela: '{win_roi}'\n- Esq: ponto. Enter: finalizar.\n- 'r': RESET. 'q'/ESC: SAIR.")
        while True:
            cv2.imshow(win_roi, current_frame_copy_for_roi)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                if len(roi_points_global) > 2:
                    roi_defined_global = True; cv2.line(current_frame_copy_for_roi, roi_points_global[-1], roi_points_global[0], (0,255,0),2)
                    cv2.imshow(win_roi, current_frame_copy_for_roi); print("ROI definida:", roi_points_global); print("Pressione tecla para continuar."); cv2.waitKey(0); break
                else: print("Mínimo 3 pontos para ROI.")
            elif key == ord('r'):
                roi_points_global = []; current_frame_copy_for_roi = original_first_frame_processed.copy(); roi_defined_global = False
                print("Pontos ROI resetados."); cv2.imshow(win_roi, current_frame_copy_for_roi)
            elif key == ord('q') or key == 27: print("Seleção ROI cancelada."); cap.release(); cv2.destroyAllWindows(); return
        cv2.destroyWindow(win_roi)

    if not roi_defined_global or len(roi_points_global) < 3: print("ROI não definida. Encerrando."); cap.release(); cv2.destroyAllWindows(); return
    save_roi_points(roi_points_global)

    mask_roi_global = np.zeros(original_first_frame_processed.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_roi_global, [np.array(roi_points_global, dtype=np.int32)], 255)
    # viz_roi = cv2.bitwise_and(original_first_frame_processed, original_first_frame_processed, mask=mask_roi_global)
    # cv2.namedWindow("ROI Aplicada (Visualização)", cv2.WINDOW_NORMAL); cv2.imshow("ROI Aplicada (Visualização)", viz_roi)
    # h_o, w_o = original_first_frame_processed.shape[:2]; disp_h_v = 700
    # if h_o > disp_h_v: cv2.resizeWindow("ROI Aplicada (Visualização)", int(disp_h_v * (w_o/h_o)), disp_h_v)
    # print("Pressione tecla na 'ROI Aplicada' para detecção."); cv2.waitKey(0); cv2.destroyWindow("ROI Aplicada (Visualização)")
    print("\n--- Fim da Parte 1: Carregamento e Seleção da ROI ---")

    balls_detected_info, _ = detect_initial_balls(original_first_frame_processed, roi_points_global, mask_roi_global)
    if not balls_detected_info: print("Nenhuma bola detectada. Encerrando."); cap.release(); cv2.destroyAllWindows(); return
    print("\n--- Fim da Parte 2: Detecção Inicial ---")

    if not initialize_trackers(original_first_frame_processed, balls_detected_info):
        print("Falha ao inicializar trackers. Encerrando."); cap.release(); cv2.destroyAllWindows(); return
    print("\n--- Fim da Parte 3: Inicialização dos Trackers ---")

    # --- Iniciar o loop de tracking do vídeo ---
    print("\n--- Iniciando Loop de Tracking do Vídeo ---")
    run_video_tracking(cap, roi_points_global, mask_roi_global) # Passa a máscara para verificação opcional

    cap.release()
    cv2.destroyAllWindows()
    print("Script finalizado.")

if __name__ == '__main__':
    main()