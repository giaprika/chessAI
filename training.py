import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from self_play import SelfPlay
from model import AlphaZeroNet
import chess

class AlphaZeroTrainer:
    def __init__(self, model, epochs=10, batch_size=64, learning_rate=1e-4):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
    
    def train(self, game_data):
        # Prepare data for training
        states, policies, values = zip(*game_data)
        states = torch.stack(states)
        policies = torch.tensor(np.array(policies), dtype=torch.float32)
        values = torch.tensor(np.array(values), dtype=torch.float32)
        
        dataset = TensorDataset(states, policies, values)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in dataloader:
                state_batch, policy_batch, value_batch = batch
                
                self.optimizer.zero_grad()
                
                # Forward pass
                policy_pred, value_pred = self.model(state_batch)
                
                # Loss calculation
                policy_loss = self.loss_fn(policy_pred, policy_batch)
                value_loss = self.loss_fn(value_pred.squeeze(), value_batch)
                
                loss = policy_loss + value_loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss / len(dataloader)}")
    
    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
    
    def load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))

def train():
    # Khởi tạo mô hình
    model = AlphaZeroNet()
    
    # Khởi tạo Trainer
    trainer = AlphaZeroTrainer(model, epochs=10, batch_size=64)
    
    # Số lượng ván cờ để tạo dữ liệu huấn luyện
    num_games = 20
    replay_buffer = []
    max_buffer_size = 10000
    
    for game_num in range(num_games):
        print(f"Game {game_num+1}/{num_games} started.")
        # Khởi tạo SelfPlay
        board = chess.Board()
        self_play = SelfPlay(model, time_limit=1.0, board=board)
        
        game_data, game_result = self_play.play_game()
        
        if game_data:
            # Thêm vào buffer
            replay_buffer.extend(game_data)
            if len(replay_buffer) > max_buffer_size:
                replay_buffer = replay_buffer[-max_buffer_size:]  # Giữ lại phần mới nhất

            # Huấn luyện trên toàn bộ buffer
            trainer.train(replay_buffer)
            print(f"Game {game_num+1}/{num_games} finished. Result: {game_result}")
        else:
            print(f"Game {game_num+1}/{num_games} skipped. No data collected.")
        
        # Lưu mô hình sau mỗi game (tuỳ chọn)
        if (game_num + 1) % 10 == 0:
            trainer.save_model(f"model_{game_num+1}.pt")
    
    # Lưu mô hình cuối cùng
    trainer.save_model("final_model.pt")

if __name__ == "__main__":
    train()
