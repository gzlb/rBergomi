import numpy as np
import pandas as pd
import tensorflow.keras as K

from abc import ABC, abstractmethod
from dataclasses import dataclass


class AbstractModelTrainer(ABC):
    @abstractmethod
    def build_model(self) -> K.models.Sequential:
        pass

@dataclass
class CNNTrainer(AbstractModelTrainer):
    input_length: int = 100
    kernel_size: int = 20
    pool_size: int = 3
    leaky_rate: float = 0.1
    output_length: int = 1

    def build_model(self) -> K.models.Sequential:
        model = K.models.Sequential([
            K.layers.Conv1D(32, self.kernel_size, activation='linear', padding='same', input_shape=(self.input_length, 1)),
            K.layers.LeakyReLU(alpha=self.leaky_rate),
            K.layers.MaxPool1D(self.pool_size, padding='same'),
            K.layers.Dropout(rate=0.25),
            K.layers.Conv1D(64, self.kernel_size, activation='linear', padding='same'),
            K.layers.LeakyReLU(alpha=self.leaky_rate),
            K.layers.MaxPool1D(self.pool_size, padding='same'),
            K.layers.Dropout(rate=0.25),
            K.layers.Conv1D(128, self.kernel_size, activation='linear', padding='same'),
            K.layers.LeakyReLU(alpha=self.leaky_rate),
            K.layers.MaxPool1D(self.pool_size, padding='same'),
            K.layers.Dropout(rate=0.4),
            K.layers.Flatten(),
            K.layers.Dense(128, activation='linear'),
            K.layers.LeakyReLU(alpha=self.leaky_rate),
            K.layers.Dropout(rate=0.3),
            K.layers.Dense(self.output_length, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

@dataclass
class Report:
    df: pd.DataFrame
    loss_history: dict