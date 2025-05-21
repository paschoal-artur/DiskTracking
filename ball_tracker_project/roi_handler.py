# roi_handler.py
import cv2
import numpy as np
import json
import os
from config import * # Importa as configurações
Config = config() # Alias para facilitar a leitura
class ROIHandler:
    def __init__(self, frame_for_selection, config_file=Config.ROI_CONFIG_FILE):
        self.points = []
        self.frame_copy = frame_for_selection.copy()
        self.original_frame = frame_for_selection # Guardar para reset
        self.is_defined = False
        self.config_file = config_file
        self.window_name = "Selecione ROI - Esq: Ponto. Enter: Finalizar. r: Reset, q: Sair"

    def _mouse_callback(self, event, x, y, flags, param):
        if self.is_defined:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            cv2.circle(self.frame_copy, (x, y), 5, (0, 255, 0), -1)
            if len(self.points) > 1:
                cv2.line(self.frame_copy, self.points[-2], self.points[-1], (0, 255, 0), 2)
            cv2.imshow(self.window_name, self.frame_copy)

    def select_roi_interactively(self):
        """Permite ao usuário selecionar a ROI interativamente."""
        self.points = [] # Reset para nova seleção
        self.is_defined = False
        self.frame_copy = self.original_frame.copy()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        h_roi, w_roi = self.frame_copy.shape[:2]
        if h_roi > Config.WINDOW_MAX_HEIGHT:
            aspect_ratio_roi = w_roi / h_roi
            display_width_roi = int(Config.WINDOW_MAX_HEIGHT * aspect_ratio_roi)
            cv2.resizeWindow(self.window_name, display_width_roi, Config.WINDOW_MAX_HEIGHT)

        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        print(f"\n--- Instruções Seleção ROI: Janela '{self.window_name}' ---")
        print("- Clique ESQUERDO: Adicionar ponto.")
        print("- Pressione 'Enter' (após 3+ pontos): Finalizar seleção.")
        print("- Pressione 'r': RESETAR pontos.")
        print("- Pressione 'q' ou ESC: SAIR sem definir.")

        while True:
            cv2.imshow(self.window_name, self.frame_copy)
            key = cv2.waitKey(1) & 0xFF

            if key == 13:  # Enter
                if len(self.points) > 2:
                    self.is_defined = True
                    cv2.line(self.frame_copy, self.points[-1], self.points[0], (0, 255, 0), 2)
                    cv2.imshow(self.window_name, self.frame_copy)
                    print("ROI definida com pontos:", self.points)
                    print("Pressione qualquer tecla na janela ROI para continuar.")
                    cv2.waitKey(0)
                    break
                else:
                    print("São necessários pelo menos 3 pontos para definir a ROI.")
            elif key == ord('r'):
                self.points = []
                self.frame_copy = self.original_frame.copy()
                self.is_defined = False
                print("Pontos da ROI resetados.")
            elif key == ord('q') or key == 27:  # 'q' ou ESC
                print("Seleção da ROI cancelada pelo usuário.")
                self.is_defined = False # Garante que não está definida
                break
        cv2.destroyWindow(self.window_name)
        return self.is_defined

    def save_roi(self):
        if self.is_defined and self.points:
            try:
                with open(self.config_file, 'w') as f:
                    json.dump(self.points, f)
                print(f"Pontos da ROI salvos em '{self.config_file}'")
            except Exception as e:
                print(f"Erro ao salvar pontos da ROI: {e}")

    def load_roi(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_points = json.load(f)
                if isinstance(loaded_points, list) and \
                   all(isinstance(p, list) and len(p) == 2 for p in loaded_points):
                    self.points = [tuple(p) for p in loaded_points]
                    self.is_defined = True
                    print(f"Pontos da ROI carregados de '{self.config_file}': {self.points}")
                    return True
                else:
                    print(f"Arquivo '{self.config_file}' não contém pontos válidos.")
            except Exception as e:
                print(f"Erro ao carregar ROI de '{self.config_file}': {e}")
        return False

    def get_mask(self, frame_shape):
        if not self.is_defined or not self.points:
            return None
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        polygon_points_np = np.array([self.points], dtype=np.int32)
        cv2.fillPoly(mask, polygon_points_np, 255)
        return mask

    def get_points(self):
        return self.points if self.is_defined else None