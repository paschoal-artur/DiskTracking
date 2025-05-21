# tracker_manager.py
import cv2
import numpy as np
import config # Importa as configurações

class TrackerManager:
    def __init__(self, tracker_type=config.TRACKER_TYPE):
        self.trackers = []
        self.tracker_type = tracker_type
        self._tracker_creation_func = self._get_tracker_creation_func(tracker_type)

    def _get_tracker_creation_func(self, tracker_name):
        if tracker_name == "CSRT":
            return cv2.TrackerCSRT_create
        elif tracker_name == "KCF":
            return cv2.TrackerKCF_create
        elif tracker_name == "MOSSE":
            try:
                return cv2.legacy.TrackerMOSSE_create
            except AttributeError:
                print("ERRO: cv2.legacy não encontrado ou TrackerMOSSE_create indisponível.")
                print("Tente 'pip install opencv-contrib-python'.")
                print("Retornando para CSRT como padrão se possível.")
                return cv2.TrackerCSRT_create # Fallback
        else:
            print(f"AVISO: Tipo de tracker '{tracker_name}' desconhecido. Usando CSRT.")
            return cv2.TrackerCSRT_create

    def initialize_trackers(self, frame, detected_balls_info):
        self.trackers = []
        if not detected_balls_info:
            print("[TrackerManager] Nenhuma bola detectada para inicializar trackers.")
            return False
        if not self._tracker_creation_func:
            print("[TrackerManager] ERRO: Função de criação do tracker não definida.")
            return False

        frame_h, frame_w = frame.shape[:2]
        initialization_successful = False

        for ball_info in detected_balls_info:
            try:
                tracker = self._tracker_creation_func()
            except Exception as e:
                print(f"[TrackerManager] ERRO ao criar tracker do tipo {self.tracker_type}: {e}")
                continue # Pula para a próxima bola

            x_init, y_init, w_init, h_init = ball_info['bbox_initial']

            # Garante que a bounding box esteja dentro dos limites e tenha tamanho > 0
            x = max(0, x_init)
            y = max(0, y_init)
            w = min(w_init, frame_w - x)
            h = min(h_init, frame_h - y)

            if w <= 0 or h <= 0:
                print(f"[TrackerManager] AVISO: BBox inválida para bola ID {ball_info['id']} após ajuste: {(x,y,w,h)}. Pulando.")
                continue
            
            adjusted_bbox = (x, y, w, h)

            try:
                success = tracker.init(frame, adjusted_bbox)
                if success:
                    self.trackers.append({
                        'id': ball_info['id'],
                        'tracker_obj': tracker,
                        'bbox': adjusted_bbox,
                        'trajectory': [(int(x + w / 2), int(y + h / 2))],
                        'color': tuple(np.random.randint(0, 255, 3).tolist()),
                        'active': True
                    })
                    initialization_successful = True
                    # print(f"[TrackerManager] Tracker ID {ball_info['id']} inicializado com BBox: {adjusted_bbox}")
                else:
                    print(f"[TrackerManager] FALHA ao inicializar tracker ID {ball_info['id']} com BBox: {adjusted_bbox}")
            except Exception as e:
                print(f"[TrackerManager] Exceção ao inicializar tracker ID {ball_info['id']}: {e}")
        
        if initialization_successful:
            print(f"[TrackerManager] {len(self.trackers)} trackers inicializados com sucesso.")
        else:
            print("[TrackerManager] Nenhum tracker foi inicializado com sucesso.")
        return initialization_successful


    def update_trackers(self, frame):
        successful_updates = 0
        for t_info in self.trackers:
            if not t_info['active']:
                continue

            success, bbox = t_info['tracker_obj'].update(frame)
            if success:
                t_info['bbox'] = bbox
                center_x = int(bbox[0] + bbox[2] / 2)
                center_y = int(bbox[1] + bbox[3] / 2)
                t_info['trajectory'].append((center_x, center_y))
                successful_updates += 1
            else:
                t_info['active'] = False
                # print(f"[TrackerManager] Tracker ID {t_info['id']} perdeu o objeto.")
        return successful_updates

    def get_tracked_objects_info(self):
        return [t for t in self.trackers if t['active']] # Retorna apenas os ativos
    
    def get_all_objects_info(self): # Para desenhar trajetórias mesmo se inativo
        return self.trackers