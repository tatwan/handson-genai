# Knowledge Distillation Training Loop Explained

This document explains how the **teacher–student distillation loop** in this notebook works, and how the teacher’s predictions are used to train the student model.

------

## High-level  idea

In this notebook, the **teacher** is a DistilBERT model already fine-tuned on SST-2, while the **student** starts as an unfine-tuned DistilBERT. The goal of knowledge distillation is to train the student to:

- Match the **teacher’s soft predictions** (logits / probabilities)
- Still perform well on the **original hard labels** (0/1 sentiment labels)

This is done by combining two losses for the student:
$$
L_{\text{total}}=\alpha \times L_{\text{distill}}+(1−α) \times L_{\text{student}}
$$


where:

- $L_{\text{distill}}$: KL divergence between student and teacher **soft** distributions
- $L_{\text{student}}$: cross-entropy between student logits and **hard** labels
- $\alpha$: weight that controls how much to trust the teacher vs. the ground truth labels

------

## Distillation loss module

The `DistillationLoss` class encapsulates this combined loss:

```python
class DistillationLoss(nn.Module):
    """
    Distillation loss combining soft label matching and hard label classification.

    L_total = alpha * L_distill + (1 - alpha) * L_student
    """
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, labels):
        # 1. Soft labels from teacher and student with temperature scaling
        # We divide logits by T to soften the distribution before softmax
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)

        # 2. Distillation loss: KL divergence between soft distributions
        # We scale by T^2 to counteract gradient scaling from temperature
        distill_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)

        # 3. Student loss: standard classification on hard labels
        student_loss = self.ce_loss(student_logits, labels)

        # 4. Combined weighted loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss
        return total_loss, distill_loss, student_loss
```

Key points:

- **Temperature T**: Higher values (e.g., 4) produce softer probability distributions and reveal more “dark knowledge” about class relationships.
- **KL divergence** (`kl_loss`): Encourages the student’s probability distribution to match the teacher’s distribution.
- **Cross entropy** (`ce_loss`): Ensures the student still predicts the correct ground truth label.

------

## Teacher vs. student in `train_epoch`

The function `train_epoch` performs one full epoch of distillation training over the student model:

```python
def train_epoch(
    student_model,
    teacher_model,
    train_loader,
    optimizer,
    scheduler,
    criterion,
    device,
):
    """Train for one epoch."""
    student_model.train()
    total_loss = 0.0
    total_distill_loss = 0.0
    total_student_loss = 0.0

    progress_bar = tqdm(train_loader, desc="Training")

    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # 1. Teacher inference (no gradients)
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            teacher_logits = teacher_outputs.logits

        # 2. Student inference (with gradients)
        student_outputs = student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        student_logits = student_outputs.logits

        # 3. Compute distillation + student loss
        loss, distill_loss, student_loss = criterion(
            student_logits,
            teacher_logits,  # ← Teacher output passed here
            labels,
        )

        # 4. Backpropagation (student only)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 5. Logging
        total_loss += loss.item()
        total_distill_loss += distill_loss.item()
        total_student_loss += student_loss.item()

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            distill=f"{distill_loss.item():.4f}",
        )

    return {
        "total_loss": total_loss / len(train_loader),
        "distill_loss": total_distill_loss / len(train_loader),
        "student_loss": total_student_loss / len(train_loader),
    }
```

## Step-by-step:

1. **Teacher inference (frozen)**
   - `teacher_model` is always in `eval()` mode and wrapped in `torch.no_grad()`.
   - Only **forward pass**, no gradients, no parameter updates.
   - Output: `teacher_logits` for the current batch.
2. **Student inference (trainable)**
   - `student_model.train()` enables gradients.
   - Output: `student_logits` for the same batch.
3. **Loss computation**
   - `criterion` is the `DistillationLoss` module.
   - Receives both `student_logits` and `teacher_logits`, plus hard labels.
   - Returns:
     - `loss`: full combined loss (used for backprop)
     - `distill_loss`: KL divergence part (student vs teacher)
     - `student_loss`: cross-entropy part (student vs labels)
4. **Backpropagation and updates**
   - `loss.backward()` computes gradients **only w.r.t. student parameters**.
   - `optimizer.step()` and `scheduler.step()` update the student model.
   - The teacher remains completely unchanged throughout training.

------

## Evaluation function

Evaluation is done with a generic helper that runs the **given** model on the validation loader:

```python
def evaluate_model(model, val_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            predictions = torch.argmax(outputs.logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total
```

This function is used to:

- Measure **teacher** accuracy (upper bound)
- Measure **student** accuracy before distillation
- Measure **student** accuracy after each epoch of distillation

------

## Training script with distillation

The outer training loop looks like this:

```python
history = []

print("Starting Distillation Training")
print("-" * 50)

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 30)

    # Train one epoch with distillation
    train_metrics = train_epoch(
        student_model,
        teacher_model,
        train_loader,
        optimizer,
        lrscheduler,
        distill_criterion,
        device,
    )

    # Evaluate student after this epoch
    student_acc = evaluate_model(student_model, val_loader, device)

    history.append(
        {
            "epoch": epoch + 1,
            "loss": train_metrics["total_loss"],
            "accuracy": student_acc,
        }
    )

    print(f"Loss: {train_metrics['total_loss']:.4f}")
    print(f"- Distillation Loss: {train_metrics['distill_loss']:.4f}")
    print(f"- Student Loss: {train_metrics['student_loss']:.4f}")
    print(f"Validation Accuracy: {student_acc:.2f}")
    print("-" * 50)

print("Distillation Complete!")
print("-" * 50)
```

Empirically in this notebook:

- Teacher accuracy ≈ 90.8%
- Student before distillation ≈ 52.2%
- Student after distillation ≈ 86.4%

So the student recovers most of the teacher’s performance while being trained to mimic its behavior.

------

## Mental model to keep in mind

When reading or modifying the code, keep this mental model:

- **Teacher = fixed oracle**
  - Provides rich, soft supervision via its logits
  - Frozen: `eval()` + `torch.no_grad()`
- **Student = trainable mimic**
  - Sees the same input and tries to both:
    - Match the teacher’s probability distribution (KL / distillation loss)
    - Fit the ground truth labels (cross-entropy loss)
- **DistillationLoss = bridge**
  - This is where the **teacher’s outputs** actually influence the student.
  - If you change `alpha` or `temperature`, you change how much and how the student listens to the teacher.

You can experiment with:

- Different `temperature` values (e.g., 1, 2, 4, 8)
- Different `alpha` values (e.g., 0.3, 0.5, 0.7, 0.9)
- More or fewer epochs

and observe how the validation accuracy and the two loss components evolve over training.