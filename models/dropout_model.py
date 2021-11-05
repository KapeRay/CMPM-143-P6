from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers

# Define your dropout model here
model = models.Sequential()

# Then, call model.compile()
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc']
)

# Train this compiled model by modifying basic_train 
# to import this model, then run:
#   python train.py