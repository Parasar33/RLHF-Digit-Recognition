import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import copy
import json
from datetime import datetime

def get_script_directory():
    return os.path.dirname(os.path.abspath(__file__))

class DigitRecognizer(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(DigitRecognizer, self).__init__()
        self.flatten = nn.Flatten()
        
        # Improved architecture with batch normalization and dropout
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return F.softmax(x, dim=1)

class DigitRecognizerApp:
    def __init__(self):
        pygame.init()
        self.setup_display()
        self.setup_colors()
        self.setup_grid_parameters()
        self.setup_model()
        self.setup_font()
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
        self.data_file = os.path.join(get_script_directory(), 'correction_data.json')
        self.load_correction_data()

    def setup_display(self):
        self.WIDTH, self.HEIGHT = 800, 600
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Digit Recognizer")

    def setup_colors(self):
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (128, 128, 128)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)

    def setup_grid_parameters(self):
        self.GRID_SIZE = 28
        self.CELL_SIZE = 16
        self.GRID_WIDTH = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_HEIGHT = self.GRID_SIZE * self.CELL_SIZE
        self.GRID_X = (self.WIDTH - self.GRID_WIDTH) // 2
        self.GRID_Y = self.HEIGHT - self.GRID_HEIGHT - 50

        self.BOX_SIZE = 50
        self.BOX_GAP = 15
        self.BOXES_X = (self.WIDTH - (self.BOX_SIZE * 10 + self.BOX_GAP * 9)) // 2
        self.BOXES_Y = 50

    def setup_font(self):
        try:
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
        except Exception as e:
            print(f"Error loading fonts: {str(e)}")
            self.cleanup()

    def setup_model(self):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")

            self.model = DigitRecognizer().to(self.device)
            weights_path = os.path.join(get_script_directory(), 'best_model.pth')
            
            if os.path.exists(weights_path):
                self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
                print("Model weights loaded successfully")
            else:
                print("Warning: No model weights found. Using untrained model.")

            self.model.eval()

        except Exception as e:
            print(f"Error setting up model: {str(e)}")
            self.cleanup()

    def load_correction_data(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    self.correction_data = json.load(f)
            else:
                self.correction_data = []
        except Exception as e:
            print(f"Error loading correction data: {str(e)}")
            self.correction_data = []

    def save_correction(self, correct_digit):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data = {
                'timestamp': timestamp,
                'grid': self.grid.tolist(),
                'correct_digit': correct_digit
            }
            self.correction_data.append(data)
            
            with open(self.data_file, 'w') as f:
                json.dump(self.correction_data, f)
            
            print(f"Saved correction: digit {correct_digit}")
            
        except Exception as e:
            print(f"Error saving correction: {str(e)}")
    def normalize_input(self, grid):
        try:
            from scipy.ndimage import gaussian_filter
            smoothed_grid = gaussian_filter(grid, sigma=0.5)
            tensor = torch.FloatTensor(smoothed_grid).reshape(1, 784).to(self.device)
            return tensor / 255.0
        except Exception as e:
            print(f"Error normalizing input: {str(e)}")
            return None

    def get_prediction(self):
        try:
            input_tensor = self.normalize_input(self.grid)
            if input_tensor is None:
                return np.zeros(10)

            with torch.no_grad():
                output = self.model(input_tensor)
            return output[0].cpu().numpy()
        except Exception as e:
            print(f"Error getting prediction: {str(e)}")
            return np.zeros(10)

    def handle_drawing(self, x, y, radius=2):
        grid_x = (x - self.GRID_X) // self.CELL_SIZE
        grid_y = (y - self.GRID_Y) // self.CELL_SIZE
        
        if 0 <= grid_x < self.GRID_SIZE and 0 <= grid_y < self.GRID_SIZE:
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    new_x = grid_x + dx
                    new_y = grid_y + dy
                    if (0 <= new_x < self.GRID_SIZE and 
                        0 <= new_y < self.GRID_SIZE):
                        distance = np.sqrt(dx**2 + dy**2)
                        if distance <= radius:
                            intensity = int(255 * (1 - distance/radius))
                            self.grid[new_y, new_x] = max(self.grid[new_y, new_x], intensity)

    def handle_correction(self):
        try:
            prompt = self.font.render("Enter correct digit (0-9):", True, self.WHITE)
            self.screen.blit(prompt, (self.WIDTH//2 - 150, self.HEIGHT//2))
            pygame.display.flip()
            
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key >= pygame.K_0 and event.key <= pygame.K_9:
                            correct_digit = event.key - pygame.K_0
                            self.save_correction(correct_digit)
                            waiting = False
                        elif event.key == pygame.K_ESCAPE:
                            waiting = False
                    elif event.type == pygame.QUIT:
                        self.cleanup()

        except Exception as e:
            print(f"Error handling correction: {str(e)}")

    def draw_grid(self):
        grid_background = pygame.Rect(
            self.GRID_X - 5, 
            self.GRID_Y - 5, 
            self.GRID_WIDTH + 10, 
            self.GRID_HEIGHT + 10
        )
        pygame.draw.rect(self.screen, self.GRAY, grid_background)

        for y in range(self.GRID_SIZE):
            for x in range(self.GRID_SIZE):
                color = (int(self.grid[y, x]), int(self.grid[y, x]), int(self.grid[y, x]))
                rect = pygame.Rect(
                    self.GRID_X + x * self.CELL_SIZE,
                    self.GRID_Y + y * self.CELL_SIZE,
                    self.CELL_SIZE,
                    self.CELL_SIZE
                )
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.GRAY, rect, 1)

    def draw_prediction_boxes(self, predictions):
        max_prob_idx = np.argmax(predictions)
        
        for i in range(10):
            x = self.BOXES_X + i * (self.BOX_SIZE + self.BOX_GAP)
            rect = pygame.Rect(x, self.BOXES_Y, self.BOX_SIZE, self.BOX_SIZE)
            
            if i == max_prob_idx:
                pygame.draw.rect(self.screen, self.GREEN, rect, 3)
            else:
                pygame.draw.rect(self.screen, self.WHITE, rect, 2)
            
            number = self.font.render(str(i), True, self.WHITE)
            prob = self.small_font.render(f"{predictions[i]:.3f}", True, self.WHITE)
            
            number_rect = number.get_rect(center=(x + self.BOX_SIZE//2, 
                                                self.BOXES_Y + self.BOX_SIZE//2))
            prob_rect = prob.get_rect(center=(x + self.BOX_SIZE//2, 
                                            self.BOXES_Y + self.BOX_SIZE + 20))
            
            self.screen.blit(number, number_rect)
            self.screen.blit(prob, prob_rect)

    def draw_instructions(self):
        instructions = [
            "Left Click: Draw",
            "C: Clear",
            "R: Record Correction",
            "ESC: Quit"
        ]
        y_offset = self.HEIGHT - 100
        for instruction in instructions:
            text = self.small_font.render(instruction, True, self.WHITE)
            self.screen.blit(text, (10, y_offset))
            y_offset += 20

    def cleanup(self):
        pygame.quit()
        sys.exit(1)

    def run(self):
        running = True
        drawing = False
        clock = pygame.time.Clock()

        while running:
            clock.tick(60)
            self.screen.fill(self.BLACK)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        drawing = True
                        x, y = event.pos
                        self.handle_drawing(x, y)
                    
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        drawing = False
                
                elif event.type == pygame.MOUSEMOTION and drawing:
                    x, y = event.pos
                    self.handle_drawing(x, y)
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE))
                    elif event.key == pygame.K_r:
                        self.handle_correction()
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            self.draw_grid()
            predictions = self.get_prediction()
            self.draw_prediction_boxes(predictions)
            self.draw_instructions()
            pygame.display.flip()

        self.cleanup()

if __name__ == "__main__":
    try:
        app = DigitRecognizerApp()
        app.run()
    except Exception as e:
        print(f"Application error: {str(e)}")
        pygame.quit()
        sys.exit(1)