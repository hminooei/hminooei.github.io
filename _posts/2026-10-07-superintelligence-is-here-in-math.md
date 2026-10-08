# Superintelligence Is Here (in Math and Code), and Practice Is How It Got Here

Last night OpenAI dropped [722 math manuscripts](https://github.com/openai/math), organized into 372 families of results, all produced by an unreleased internal model. My reaction on X was short:

> SI (super intelligence) concretely shown (already achieved but some had doubts) in math. The same is true for coding. One caveat: the SI is jagged, which I consider it as rough corner that soon will be smooth.
>
> — [@sardaaroo](https://x.com/sardaaroo/status/2107716178672050413)

Here's the longer version.

## 📚 What Was Released

According to the [repo's README](https://github.com/openai/math):

- The model was posed **~4,000 open research problems**, and the strongest results were collected into **722 manuscripts**.
- Each result took on average about **three hours of ChatGPT Pro thinking compute**.
- Many (not all) of the results come with **Lean formalizations**, which are machine-checked proofs. More are promised.
- Abridged **reasoning summaries** were released for highlights such as the irrationality exponent of π, the Mahler conjectures, Kaplansky's direct-finiteness conjecture in characteristic two, and the isomorphism of free group factors.
- The headline-grabber is a zero-free half-plane for Dirichlet L-functions (Re s > 7/8), which the internet quickly called the ["quasi-Riemann hypothesis"](https://www.kucoin.com/news/flash/openai-solves-722-math-problems-quasi-riemann-hypothesis-proven). To be clear, it's *not* the Riemann hypothesis. It's still a big deal.

One line in the README stood out to me more than any single theorem:

> *"We expanded these evaluations after performance on our existing mathematical evaluations saturated."*

In plain words, the model ran out of exam questions, so they handed it open research problems.

## 🗣️ The Reactions

The excitement was immediate:

- Crémieux [wrote](https://x.com/cremieuxrecueil/status/2107600945383112990): *"This 7/8 quasi-Riemann hypothesis proof looks like one of the most important advances in analytical number theory in decades... Plus 3,998 more things! Oh my god."*

A few more reactions that stood out to me:

<blockquote class="twitter-tweet" data-dnt="true"><a href="https://twitter.com/alexkontorovich/status/2107609087902941646">Alex Kontorovich on X</a></blockquote>

<blockquote class="twitter-tweet" data-dnt="true"><a href="https://twitter.com/nechita_ion/status/2107679271108116880">Ion Nechita on X</a></blockquote>

<blockquote class="twitter-tweet" data-dnt="true"><a href="https://twitter.com/obhisheksaha/status/2107801342722928899">Obhishek Saha on X</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

Thousands of research problems were attempted, hundreds of results came out, and many come with machine-checked proofs. Debating whether models can do research-level math is over.

## 🧗 "Jagged" Superintelligence

The caveat in my tweet matters. This superintelligence is **jagged**. The same model that proves new results about L-functions can still fumble tasks a teenager finds trivial.

I see that jaggedness as **rough corners, not a ceiling**. To see why, look at *how* these models got so good at math and code in the first place.

## 🎯 Practice Makes Perfect

How does a human get great at anything, whether piano, chess, surgery, or lifting?

**Practice with feedback.** Lots of attempts, a clear signal about what went right and wrong, adjust, repeat. Then raise the difficulty and do it again. Nobody becomes a grandmaster by reading about chess.

Models are getting superhuman the same way. Through **reinforcement learning (RL)**, a model attempts a task, gets scored on the result, and is nudged toward whatever worked. It does this millions of times.

Initially, math and code went first for a simple reason: **their feedback was exceptionally clean.**
- In code, the tests pass or they don't.
- In math, [Lean](https://lean-lang.org/) (a programming language and proof checker that verifies every logical step of a proof by computer) accepts the proof or it doesn't.

That's no longer the whole story. Today's strong models don't need execution-based feedback like running tests or compiling a proof to get a useful signal. **Another model can judge the work**: reading a proof or a pull request the way a seasoned reviewer would, spotting gaps, and grading the quality of the reasoning. The coach no longer has to be a stopwatch. It can be an expert.

When feedback is precise and cheap, you can run an enormous number of high-quality practice reps, and quality reps compound. That README line about evaluations "saturating" is what happens when a student gets so good at practice problems that you have to hand them unsolved ones.

This also explains the jaggedness. Skills with fuzzy, slow, or expensive feedback (taste, long-horizon judgment, messy real-world tasks) haven't had as many good practice reps *yet*. As labs build better ways to practice and grade those skills, the rough corners will get sanded down. They're the same corners, filled in by the same practice process.

## 🏋️ Where Does a Model Practice? In a Gym

The place a model does those reps is called an **RL gym** (or RL environment). If you're not sure what that means, I wrote a companion post that explains it with actual gyms and dumbbells: **[What Is an RL Gym? (Explained With Actual Dumbbells)](/2026/10/07/what-is-an-rl-gym.html)**.

The short version: tasks are the dumbbells, attempts are the reps, the grader is the coach, and progressive overload is how you keep getting stronger. Math and code had the best-equipped gyms in town, which is why they got strong first.

## 🔮 What's Next

1. **Math and coding** are now in superhuman territory for a growing range of problems, and the gap will widen as harder environments are built.
2. **The jagged edges will smooth out** as gyms are built for skills that currently lack clean feedback.
3. **Verification and attribution become the bottleneck.** When a model can produce 722 manuscripts, the scarce resource is no longer proofs. It's trust in them. Formal verification (Lean) and clear credit norms need to scale as fast as the models do.

Practice makes perfect. We now have systems that can practice at a scale no human ever could.
