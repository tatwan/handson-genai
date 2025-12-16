#!/usr/bin/env python3
"""Script to add solution cells to the neural network basics notebook."""

import json

# Read the notebook
with open('01_neural_network_basics.ipynb', 'r') as f:
    nb = json.load(f)

# Find the index of the student challenge code cell
challenge_idx = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        if 'Student Challenge: Build an improved model' in source:
            challenge_idx = i
            break

if challenge_idx is None:
    print('Could not find challenge cell')
    exit(1)

# Create solution markdown cell
solution_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '---\n',
        '\n',
        '### 💡 Solution: Improved MNIST Classifier\n',
        '\n',
        'Below is one possible solution that incorporates several best practices:\n',
        '\n',
        '| Technique | What It Does | Why It Helps |\n',
        '|-----------|-------------|-------------|\n',
        '| **More layers** | Deeper network (3 hidden layers) | Learns more complex patterns |\n',
        '| **Batch Normalization** | Normalizes layer inputs | Faster training, better stability |\n',
        '| **Dropout** | Randomly drops neurons during training | Prevents overfitting |\n',
        '| **More neurons** | 256 → 128 → 64 architecture | More capacity for learning |\n',
        '| **More epochs** | 15 epochs instead of 10 | More training iterations |'
    ]
}

# Create solution code cell 1 - model definition
solution_code1 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# ===== SOLUTION: Improved MNIST Classifier =====\n',
        '\n',
        '# Build an improved model with best practices\n',
        'improved_model = keras.Sequential([\n',
        '    layers.Input(shape=(784,)),\n',
        '    \n',
        '    # Layer 1: More neurons for richer initial representation\n',
        '    layers.Dense(256, activation="relu"),\n',
        '    layers.BatchNormalization(),  # Stabilizes and speeds up training\n',
        '    layers.Dropout(0.3),          # Prevents overfitting (30% dropout)\n',
        '    \n',
        '    # Layer 2: Gradual reduction in size\n',
        '    layers.Dense(128, activation="relu"),\n',
        '    layers.BatchNormalization(),\n',
        '    layers.Dropout(0.3),\n',
        '    \n',
        '    # Layer 3: Continue narrowing\n',
        '    layers.Dense(64, activation="relu"),\n',
        '    layers.BatchNormalization(),\n',
        '    layers.Dropout(0.2),          # Less dropout near output\n',
        '    \n',
        '    # Output layer: 10 classes (digits 0-9)\n',
        '    layers.Dense(10, activation="softmax")\n',
        '])\n',
        '\n',
        '# Compile with Adam optimizer\n',
        'improved_model.compile(\n',
        '    optimizer="adam",\n',
        '    loss="sparse_categorical_crossentropy",\n',
        '    metrics=["accuracy"]\n',
        ')\n',
        '\n',
        '# Display the improved architecture\n',
        'print("📊 Improved Model Architecture:")\n',
        'improved_model.summary()\n',
        'print(f"\\n💡 Total parameters: {improved_model.count_params():,}")\n',
        'print(f"   (vs ~109,000 in original model)")'
    ]
}

# Create solution code cell 2 - training
solution_code2 = {
    'cell_type': 'code',
    'execution_count': None,
    'metadata': {},
    'outputs': [],
    'source': [
        '# Train the improved model\n',
        'print("🎓 Training improved model...\\n")\n',
        '\n',
        'improved_history = improved_model.fit(\n',
        '    X_train_flat, y_train,\n',
        '    epochs=15,            # More epochs for better convergence\n',
        '    batch_size=32,\n',
        '    validation_split=0.1,\n',
        '    verbose=1\n',
        ')\n',
        '\n',
        '# Evaluate on test set\n',
        'test_loss, test_accuracy = improved_model.evaluate(X_test_flat, y_test, verbose=0)\n',
        '\n',
        'print(f"\\n✅ Improved Model Results:")\n',
        'print(f"   Test Accuracy: {test_accuracy:.2%}")\n',
        'print(f"   Test Loss: {test_loss:.4f}")\n',
        'print(f"\\n📈 You should see improved accuracy (likely 98%+) compared to the original ~97%!")'
    ]
}

# Create explanation markdown cell
explanation_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [
        '### 🔍 Understanding the Improvements\n',
        '\n',
        '**1. Batch Normalization**\n',
        '- Normalizes the inputs to each layer, reducing internal covariate shift\n',
        '- Allows higher learning rates and faster training\n',
        '- Acts as a mild regularizer\n',
        '\n',
        '**2. Dropout**\n',
        '- Randomly sets a fraction of neurons to 0 during training\n',
        '- Forces the network to learn redundant representations\n',
        '- Very effective at preventing overfitting\n',
        '\n',
        '**3. Deeper Architecture (3 hidden layers)**\n',
        '- More layers = more capacity to learn complex patterns\n',
        '- Gradual size reduction (256 → 128 → 64) creates a "funnel" effect\n',
        '\n',
        '> **💡 Pro Tip:** For even better results on image data, try Convolutional Neural Networks (CNNs) which can achieve 99%+ accuracy on MNIST!'
    ]
}

# Insert the new cells after the challenge cell
nb['cells'].insert(challenge_idx + 1, solution_md)
nb['cells'].insert(challenge_idx + 2, solution_code1)
nb['cells'].insert(challenge_idx + 3, solution_code2)
nb['cells'].insert(challenge_idx + 4, explanation_md)

# Write the modified notebook
with open('01_neural_network_basics.ipynb', 'w') as f:
    json.dump(nb, f, indent=4)

print('✅ Solution cells added successfully!')
print(f'   Inserted 4 new cells after the student challenge (index {challenge_idx})')
