from flax import linen as nn

class TradingModel(nn.Module):
    features: list

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)
        return nn.Dense(3)(x)  # Output Buy, Sell, Hold

model = TradingModel(features=[128, 64])  # Optional Architecture
