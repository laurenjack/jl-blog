---
title: "Evaluating the Evals"
description: "How online evaluation determines the winners in AI"
category: essay
pubDate: "2026-05-13"
---

The most interesting lesson I learned transitioning from academic ML to industry is how weakly evals predict real world performance. This well-known [result](https://chbrown.github.io/kdd-2013-usb/kdd/p1168.pdf) from Microsoft demonstrated exactly that, the conclusion being "focus more on running lots of AB rests, less on perfecting algorithms".

In my time leading a RecSys team, the biggest predictor of our revenue impact each year was how many experiments we ran. That is not to say good evals are not important, iterating on models without evals is like doing gradient descent without a gradient, they're absolutely essential. It is just that **good evals are hard**, and they ultimately need to be refined against **online data** from real customers.

Whether it comes from AB tests, data analytics, or simply conversations with customers, online data is ultimately the bottleneck. Without it, you don't know what good is, you simply can't compete. This is why I expect Anthropic and OpenAI to pull away from the rest, despite distillation. Online data is the gradient for the outermost optimization loop, and its effects on AI market structure are underrated.

## RecSys - Easy but Still Hard

You might think evals are "solved" for RecSys, after all it's just about the oldest commercial application of Machine Learning. In some sense this is true, there is a general blueprint for building recommender systems that works. There are also standardized offline evaluation metrics for ranking (e.g. [NDCG](https://www.evidentlyai.com/ranking-metrics/ndcg-metric), [MAP](https://www.v7labs.com/blog/mean-average-precision) or [ROC AUC](https://www.evidentlyai.com/classification-metrics/explain-roc-curve))[^1]. Yet even if you do everything right, with a test set crushing algorithm, you're very unlikely to win the first AB test.

First there is the question of *attribution*. Before an order the customer sees an item multiple times, which *impression*[^2] deserves credit for the order? Last click attribution is a common choice in Adtech, but building an eval based on that is going to favour re-targeting algorithms[^3], which are annoying when you're shopping.

Then you have the question of the reward signal. Clicks are easy to maximize but don't translate to revenue. Orders are the outcome we want but sparse and difficult to attribute[^4]. Add to carts are a good middle ground, but are often used more like a wishlist. Over what time period should the test set be sampled? Which ranking metric should we use? Do we correct our test set for popularity bias?

Empirically, none of these questions can be answered offline, nor can they be answered with a single AB test. Real revenue outcomes roll in slowly, you're forced to be sample efficient[^5], leaning heavily on your intuitions, experience and deductive abilities to explore eval space.

After a few years we settled on last-impression-first-click[^6] attribution, NDCG50 as a ranking metric and a linear combination of add-to-carts and orders as a reward. This works, it's a weak predictor of revenue which is what we test for in AB tests. Nonetheless, we're almost certainly at a local minimum in eval space. Each time we've tried more revenue heavy evals, we've had negative or at least null results.

E-commerce recSys is in some sense the perfect lab for evaluating evals. You've got B2C statistical power and an outcome that cleanly represents customer value. Still, searching for evals is like a harder version of searching for model architectures, you need to change multiple things to escape minima, except you have no validation set to tell you you're on the right track.

## Frontier AI Labs

The scope of a frontier AI lab is enormous, even if they were to exclusively focus on coding. Customer outcomes are fuzzy, and fuzzy outcomes translated to precise metrics are hackable. Labs need a range of online data from AB tests all the way to customer vibes to know and improve their models in the wild.

Suppose, for a few days, you had unrestricted access to the flagship model from OpenAI or Anthropic, for the sake of distillation. There is no denying that this is a huge benefit, but simultaneously it is naive to assume this allows you to offer a competing product. The chasm between offline and online performance is large. This is a hard problem for RecSys, and a much harder one for something as complex as frontier models.

The fundamental problem is that the flagship model is operating in a different environment to your own. It is optimized for a different harness, for a different UI with a different customer base. If you've never iterated on a model in production, from the outside coding models all look roughly the same. However there will be 100's of little frustrating ways that your model fails. Using tool calls that don't exist, getting stuck in reasoning loops, are some common examples.

The ultimate evaluation loop from which you discover these failures is your customer base, which runs faster the more customers you have. This optimization loop is at least an order of magnitude slower than researchers internally hill climbing evals. Despite your best efforts, you simply will not develop all evals to cover all your failure cases, you'll have to put it out there and iterate, which Open AI and Anthropic are doing faster than you are. What do you think the non researchers at these companies are working on?

The conclusion is bearish for Google, xAI and Meta, at least as far as coding is concerned. Despite being extremely well-capitalized, these companies have an eval-evaluating outer feedback loop, because they do not have a large base of software engineers using their products. As far as western companies are concerned, the only horses in the race are OpenAI, Anthropic and Cursor.

Cursor has fallen out of vogue recently, but the outer loop is in place, because they still have a large and growing enterprise business. They have the capital, compute, and *might* be able to acquire enough talent to become a lab in their own right. It is too late for Google, xAI or Meta to catch up as an enterprise coding tool on their own. . This is why xAI will likely buy Cursor, or continue their transition to a CoreWeave style business.

## Challengers

So far, this post risks being misconstrued as lab-maximalism, the idea that one or two companies will conduct all commerce in the limit. On the contrary, the argument should lead one to believe in a greater degree of decentralization as time goes on.

It is well accepted by lab-maximalists that biological and physical advances will take much longer simply because of the large number of experiments that must be run. What is underrated by these same individuals is the degree to which the progress of the AI labs is also bottlenecked by empirical customer feedback.

This means that companies that form their feedback loops now have an advantage over the labs, whose opportunity cost of pivoting their compute and from coding and the most white collar work is so large, they simply cannot put any attention to these domains. This includes more narrow specialities, voice, robotics and even continual learning itself.

Voice is an interesting example, because OpenAI does it themselves, whereas Anthropic uses ElevenLabs. OpenAI no doubt has advantages over ElevenLabs (compute, talent, starting early) but the breadth of applications where ElevenLabs receives outer loop feedback is a bullish factor for them winning.

In summary, AI is fundamentally limited by empirically obtained feedback, which is obtained at the pace at which the world around it moves. It is this feedback which ultimately makes evals good, and is ultimately the limiting factor on progress. Those who refine those feedback systems for particular domains stand to win.

[^1]: Essentially these metrics are different ways of asking the question: "Did the new algo put the things the customer actually bought higher?".

[^2]: An "impression" is the showing of an item to a customer at a given time

[^3]: Re-targeting essentially means showing items you've already clicked on. Sometimes great for ads, always annoying when you are shopping in-platform.

[^4]: There is a long time between the impression where the customer decided to buy the item and the order itself.

[^5]: This is why LLMs (circa 2026) cannot come up with good evals by themselves. They can certainly provide a good starting point, but they won't know how to improve them.

[^6]: The last impression, before the first click that happened before the add to cart / order
