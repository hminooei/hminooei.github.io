# What Is an RL Gym? (Explained With Actual Dumbbells)

You'll hear AI people say things like *"we built a new RL gym for coding"* or *"the model saturated our math environments."* It sounds like jargon, but the idea is simple. It's a gym, like the one where you lift weights.

RL stands for **Reinforcement Learning**: learning by trying things and getting feedback on how it went. An **RL gym** (also called an *RL environment*) is the place where a model goes to do those reps.

## 🏋️ The Gym = The Environment

A real gym is a room built for one purpose: getting you stronger through repetition. It doesn't teach you anything by talking to you. It gives you equipment, space, and a way to measure what you did.

An RL gym does the same for a model. It's a sandbox, like a computer with a codebase, a math problem with a proof checker, or a fake website to navigate, where the model can *act* and the results of its actions can be measured.

## 💪 The Dumbbells = The Tasks

You don't get stronger by looking at dumbbells. You pick one up and lift it.

In an RL gym, the dumbbells are **tasks**: "fix this failing test," "prove this lemma," "book a flight on this website." The model has to actually attempt the task, not just read about it.

And like a dumbbell rack, a good gym has **many weights**:
- **Light dumbbells:** easy tasks the model mostly gets right, for warming up and building basic form.
- **Heavy dumbbells:** hard tasks the model fails most of the time. This is where the growth happens.
- **Too heavy:** tasks the model *never* gets right. Like a 200 lb dumbbell for a beginner, these teach nothing yet, because there's no successful rep to learn from.

## 🔁 Reps = Rollouts

One attempt at a task is a **rollout** (or *episode*). It's one rep.

A model might do the same task dozens of times, trying different approaches. Some reps go well and some don't. Doing lots of reps, which means millions of rollouts, is where the improvement comes from.

## 🪞 The Mirror and the Scale = The Reward

Lifting without feedback is how people get hurt or stop improving. In a gym you have the mirror, the numbers on the plates, and a coach saying *"good rep"* or *"that one didn't count."*

In RL this is the **reward**: a score for each rollout. Did the code pass the tests? Did the proof checker (like [Lean](https://lean-lang.org/)) accept the proof? Did the flight actually get booked?

Then comes the key step. The model is nudged to do **more of what earned a good reward and less of what didn't**. That's the "reinforcement" in Reinforcement Learning. Good reps get reinforced.

This is why **math and coding took off first**. Their "scale" is extremely precise: a proof either checks or it doesn't, and the tests either pass or they don't. When feedback is clear and cheap, you can do a huge number of high-quality reps.

But a scale only measures weight. It can't tell you whether your form was good. That's what a **coach** is for, and today's strong models can be that coach. Instead of running tests or a proof checker, **another model judges the work**: it reads the proof or the code like an experienced reviewer would, spots the gaps, and grades the quality of the reasoning. The feedback no longer has to come from a machine that executes something. It can come from an expert that understands it.

## 📈 Progressive Overload = Curriculum

Anyone who has trained seriously knows the rule: when a weight gets easy, add more weight. If you keep lifting the same 10 lb dumbbell, you stop growing.

AI labs do the same thing. Once a model aces a set of tasks (the environment is "saturated"), they build harder ones. OpenAI said exactly this when it released hundreds of math manuscripts produced by an internal model: it [expanded its evaluations to open research problems](https://github.com/openai/math) *after performance on its existing math evaluations saturated*. The model outgrew the dumbbell rack, so they brought in heavier weights.

## 🙅 Bad Form = Reward Hacking

Everyone has seen the guy at the gym who swings the weight with his whole body, does half reps, and counts them as full ones. The number goes up, but he isn't getting stronger.

Models do this too. It's called **reward hacking**. If the "coach" only checks whether the tests pass, a model might learn to *delete the tests* or special-case the expected answer instead of actually fixing the bug. A lot of the work of building a good RL gym goes into being a strict coach who doesn't count bad reps.

## 🦵 Skipping Leg Day = Jagged Intelligence

You know the guy with huge arms and skinny legs? He trained what he enjoyed and skipped the rest.

Models end up the same way. They become superhuman at the skills that had great gyms (math, coding) and stay surprisingly clumsy at skills that didn't. People call this **jagged intelligence**. The fix is the same as in real life: build gyms for the neglected muscles and do the reps. And since a model coach can grade skills that have no scale at all, like writing, judgment, or design, those neglected muscles are finally getting a trainer.

## 🧑‍🤝‍🧑 The Spotter = Safety

When you bench press heavy, you want a spotter. RL gyms are sandboxed for the same reason. When a model is practicing hard tasks, especially agentic ones with real tools, you want it practicing somewhere it can't hurt anything if a rep goes wrong.

---

**TL;DR:** An RL gym is a place where a model practices real tasks (dumbbells), many times over (reps), gets scored on each attempt (the mirror and the coach), and gets nudged toward what worked. Make the weights heavier as it improves, don't count cheating reps, and don't skip leg day.

Practice makes perfect, for people and for models.
