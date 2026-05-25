# Thought Process — Newton's Cradle

Summary - This file documents my thought process, my problems and my learnings throughout the duration of the task.
          The journey through the task was a roller coaster 🎢, did not clearly understand what to do initially, it
          was really frustrating 😡. But after lots of mistakes and clarifications later, here it is 🥳.
---

## 1. First Reading of the Task

When I first read the task, I thought the task was simply to make a model which predicts one of the given 
actions like collision (which is used, because i thought it was easiest to model, at that time). 
Two objects are in the scene, will they collide or not? Binary label, binary classifier. That felt
like a solvable problem.

## Online Data Search

Then I started searching online for the available datasets which can be used in my task.
I landed upon the CLEVRER dataset (http://clevrer.csail.mit.edu/). It involved multiple 3D objects having multiple collisions.
Their research paper answered different questions. (Source - https://arxiv.org/pdf/1910.01442)
Then I realised it would be very difficult to model something with multiple collisions, so i tried to find other datasets.
But finally i decided to generate my own 2D dataset using "pymunk" library that i would use to train my model.

## 2. Simple Model Task - Predict Collision
I started building toward that. A pymunk simulation with two objects, labeled
collision=1 or collision=0. The dataset only had 2 objects travelling in linear trajectories in random
directions and the model output was a single binary label.

The model trained. The accuracy was reasonable. And then I hit a wall: I had
no idea how to explain that the model had learned physics. Collision prediction
is a geometry problem — if two objects are moving toward each other, they will
probably collide. A model could solve this by learning "are the objects moving
toward each other?" without understanding any physics at all.

---

## 3. Making the Task More Complex — The Multi-Family Approach

I actually confused what is it meant by "model understands physics dynamics".
My solution to the previous problem was to make the task harder. If I added diverse
physics families — gravitational attraction, orbital dynamics, spring forces,
mass asymmetry, spinning objects — and the model could correctly classify which
family it was looking at, surely that would mean it understood physics?

I designed a 5-family dataset:
- Family 1: Force-free momentum (straight line collisions and near-misses)
- Family 2: Collision elasticity (elastic, inelastic, sticky)
- Family 3: Force-based interactions (gravitational, orbital, spring, repulsion)
- Family 5: Mass asymmetry (light hits heavy, heavy hits light, equal mass)
- Family 6: Spinning dynamics

The model would have two output heads: collision prediction and physics family
classification. I thought: if the latent space has one cluster for gravitational
dynamics, another for spinning dynamics, another for elastic collisions, that
would prove the model understood the different types of physics.

I spent significant time designing this — the boilerplate, the FAMILY_TARGETS
structure, the discard conditions, the metadata CSV. I partially implemented
the dataset generation script.

---

### Realization of my mistake

While working on the multi-family approach, I ran an experiment. I trained a
model on the family classification task, extracted the latent vectors, and
plotted the latent space. The clusters for each physics family were reasonably
separated — it looked like it was working.

Then I did a more careful diagnostic. I took just one physics family and colored
the latent space points by the initial velocity of the objects in each sample.
If the model truly understood the physics of that family, similar initial
velocities should produce similar latent vectors — the model should encode
"how fast are these objects moving" as a meaningful feature.

But i found out the points were randomly scattered. There was no relationship between initial velocity
and position in the latent space. The colors were distributed completely
randomly across the cluster.

This told me something important: the model had learned to recognize the visual
appearance of each physics family, not the underlying physics. Gravitational
trajectories curve — the model learned "curved trajectory = Family 3." Spinning
objects look different from non-spinning ones — the model learned that visual
distinction. But it had not learned anything about the actual physical quantities
involved: velocity, force, mass, momentum.

Family clustering does not mean physics understanding. It means visual pattern
recognition across categories that happen to correspond to physics families.
A model that clusters "gravitational" samples together might just be recognizing
curved lines. It does not mean the model knows that the curvature comes from an
inverse-square force law, or that the curvature radius is determined by the
masses and initial velocity.

I had made the task more complex without making it more physically meaningful.

---

## 4. What Physics Understanding Actually Means

This forced me to think more carefully about what the task was actually asking.

"A latent space descriptive enough to understand the underlying dynamics" does
not mean a latent space that separates physics categories. It means a latent
space where the geometry corresponds to physical quantities. If two scenes
have similar physics, they should be nearby in latent space — not because they
look the same, but because the physical quantities governing them are similar.

The key question is: can you read physical variables off the latent
representation? Can you find a direction in latent space that corresponds to
velocity? To mass? To elasticity? If yes, the model understood physics. If the
latent space is organized by visual appearance rather than physical quantities,
the model learned a visual classifier, not a physics model.

I had been checking the wrong thing. The family clustering test was asking
"did the model separate the categories?" The right test is "does the latent
space encode the continuous physical quantities within and across those
categories?"

---

## 5. Why I Pivoted to CoR Prediction

I re-read the task PDF more carefully:
> you can choose an output not listed above as well, but you would have to give a reasonable explanation as to why this output serves our purpose of instilling physical reasoning into the model

So after multiple clarifications later 😊
I landed on something that might satisfy the conditions of the task.
Predicting the coefficient of restitutiton (CoR) of a collision.
Instead of predicting collision, I changed to predicting the dynamics of collision.
CoR is a specific,measurable physical quantity. Its equation is:

    CoR = |v_b_after - v_a_after| / |v_a_before - v_b_before|

If a model predicts CoR correctly, it has learned something about the ratio of
relative velocities before and after collision. That is a physical relationship,
not a visual category.

More importantly, CoR prediction makes the latent space test meaningful. If the
latent space encodes CoR, I can show:
- Pearson correlation between latent dimensions and true CoR values
- t-SNE colored by CoR — does the gradient make physical sense?
- Interpolation between CoR=0.0 and CoR=1.0 in latent space — does walking
  through the latent space produce physically valid intermediate states?
- Scene retrieval — does nearest neighbour in latent space mean nearest in CoR?

These tests directly answer "does the latent space encode the physical
quantity?" rather than "does the latent space separate visual categories?"

The multi-family approach could not produce these tests cleanly. With 5 families
and binary collision labels, the physical quantities were entangled. With a
single controlled experiment varying only CoR, I could demonstrate a direct
relationship between latent geometry and a specific physical law.

---

## 6. Dataset Design — The Controlled Experiment

The final dataset is a controlled experiment. Everything is held constant except
CoR:
- Two equal-mass squares (removes momentum asymmetry)
- Head-on horizontal collisions only (removes angular components)
- No rotation, no gravity, no damping (removes confounding physics)
- CoR varies from 0.0 to 1.0 in steps of 0.1 (11 bins, 300 samples each)
- 3300 total samples stored as MP4 videos at 512×512

Every constraint is a deliberate choice. I am not hiding complexity — I am
isolating the variable I want to study. 

One thing I did not anticipate: loading MP4 files on-the-fly during training
caused the DataLoader to stall for very long with no output. The fix was
to preprocess all 3300 videos into .npz files (20 frames, 64×64) before
training. This reduced per-epoch time from to about 28 seconds.
I only discovered this by running the code and waiting. It was a real bottleneck
that costed time.

---

## 7. Loss Function — Encoding Physics as a Constraint

The first version of the model predicted only CoR. But predicting a number that
correlates with CoR is not the same as understanding the CoR equation.

I added two velocity prediction heads — final velocity of object A and B. This
forced the encoder to learn momentum transfer, not just "bounciness."

The most important addition was the physics consistency loss:

    predicted_cor_from_velocities = |vel_b_pred - vel_a_pred| / |vel_a_init - vel_b_init|
    consistency_loss = MSE(predicted_cor_from_velocities, cor_pred)

This term penalizes the model when its velocity predictions and its CoR
prediction contradict each other. A model that predicts CoR=0.8 but also
predicts both objects moving in the same direction at the same speed is
violating the CoR equation — and the consistency loss punishes it.

The model was never told this equation explicitly. The consistency loss enforced
it as a training constraint. This is the most physically motivated part of the
entire design.

---

## 8. Architecture Choices and What Happened

I tested 4 architectures, all with the same three output heads:

**Architecture 1 (CNN + LSTM):** Baseline. CNN extracts spatial features per
frame, LSTM processes the sequence. Sequential — each timestep depends on the
previous..

**Architecture 2 (CNN + Transformer):** Self-attention replaces LSTM. Every
frame can directly attend to every other frame without sequential dependency.

**Architecture 3 (3D CNN):** Treats the video as a spatiotemporal volume. 3D
kernels detect patterns across space and time simultaneously.

**Architecture 4 (ResNet18 + LSTM):** Transfer learning approach.
A pretrained ResNet18 acts as the spatial feature extractor for each frame,
paired with an LSTM to process the temporal sequence.

---

## 9. Results That Surprised Me

**Architecture 3 winning.** The simplest temporal mechanism won. I think
3D convolutions detect the collision as a local spatiotemporal event — the
moment of contact in space-time — without needing to carry information across
timesteps the way LSTM does or compute global attention the way Transformer does.
Simple and local was better than sequential or global for this specific task.

**Architecture 4 failing OOD catastrophically.** R²=-1.15 means it predicted
worse than always guessing the mean. A result that is worse than the trivial
baseline is not just wrong — it is actively misleading. The model found a
shortcut that worked perfectly in-distribution and failed completely outside it.
This is a warning about testing only on the distribution you trained on.

**Momentum conservation holding at 7.16%.** I never trained on Newton's law of
momentum conservation. The physics consistency loss only enforces the CoR
equation. Yet the model's velocity predictions satisfy momentum conservation
with 7.16% relative error on unseen test samples. The velocity supervision
loss implicitly steered toward momentum-conserving predictions because the
ground truth velocities satisfy it — but the model had to generalize this to
new samples it had never seen, and it did.

**Scene retrieval at 87.4% improvement over random.** The nearest neighbour
in 256-dimensional latent space is the nearest neighbour in CoR 89.1% of the
time. This is the most direct answer to "does the latent space encode physics?"
It does not require any statistical argument. You show a query video and the 3
retrieved videos — they have different sizes, speeds, and positions, but nearly
identical CoR values. The model found physically similar scenes, not visually
similar ones.

---

## 10. Dilemma - Building robust model or a research oriented approach
After completing the CoR prediction pipeline and seeing strong in-distribution results, I had a concern. 
The model was trained on a very specific setup: equal-mass squares, horizontal head-on collisions, 
fixed canvas size, controlled lighting (white background, black objects). It had never seen collisions at different angles, different mass ratios, different object shapes, or different visual conditions.
The OOD tests on unequal masses and off-centre collisions helped, but the model was still far from a general-purpose physics engine.

This raised a question I could not answer myself: was I building the wrong thing? Was the task asking for a robust model that works across all physical configurations — or was it asking for a research experiment that demonstrates physics understanding within a controlled setting?
So I asked for a clarification - 
> Is our task to build a robust model or is our task research oriented — test, infer, and explain — and not necessarily building a high-grade model?
> Ans - The aim of the task would be closer to the latter. We don't need a 100% robust model that works for all types of orientation, lighting, resolutions and what not. What we value more is the decisions you made throughout this task. What output did you choose and why, what models and architectures did you try and why, and why they failed or succeeded. The task is not about the accuracy of the final model but rather how you approached and iterated throughout.

## 10. What I Would Do Differently

**Start with CoR.** The multi-family approach was an attempt to make the task
harder in the hope that harder would mean more meaningful. It was the wrong
kind of harder. CoR prediction is simpler to set up but produces more honest
evidence of physics understanding. I should have read the task PDF more
carefully at the start — the CoR example was there from the beginning.

**Do the diagnostic experiment earlier.** The latent space coloring by initial
velocity was the experiment that told me the multi-family approach was not
working. I should have done this diagnostic much earlier, before investing
significant time in the dataset design.

**Preprocess the dataset before any training.** The MP4 loading bottleneck
cost real time. Any future vision project should preprocess data to a fast
format (npz, lmdb, hdf5) before writing any training code.

**Test OOD from the start.** I added OOD testing late in the project. In
hindsight it should have been part of the evaluation plan from the beginning.
The OOD results are the most informative results in the entire project —
they reveal which models actually learned physics and which memorized the
training distribution.

---

## 11. What I Learned

**The bottleneck is almost never the model.** The training speed problem was
the DataLoader, not the GPU. The most impactful fix was a preprocessing script,
not an architecture change.

**Clustering is not understanding.** This was the central lesson. A model that
separates physics categories in latent space has learned to recognize visual
patterns associated with those categories. It has not necessarily learned the
physical quantities that determine the dynamics. The right test is whether
physical quantities are encoded in the latent geometry — not whether visual
categories are separated.

**Failure is informative if you understand why.** Architecture 4's OOD failure
told me more about what the other architectures actually learned than any
accuracy number. A model that fails in a predictable way — because it memorized
a specific physical setup — is evidence that the other models which did not
fail were doing something genuinely different.

**The loss function encodes physical assumptions.** The physics consistency loss
was the design decision that mattered most. It is the only part of the training
that explicitly connects the model's outputs to a physical law. Everything else
— the architecture, the number of frames, the resolution — was secondary.

**Evidence should be diverse and independent.** The strongest part of this
project is that physics understanding is demonstrated through 8 independent
tests: prediction accuracy, latent correlation, t-SNE gradient, interpolation,
causal intervention, OOD generalization, momentum conservation, and scene
retrieval. Any one of these could be a coincidence or an artifact. All eight
pointing in the same direction is harder to dismiss.
