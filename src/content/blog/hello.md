---
title: 'Hello, World'
description: 'Smoke test for math and code rendering.'
pubDate: '2026-05-16'
---

This is a smoke-test post. If you can read it on the blog, the site builds. If the equation below is rendered as math (not bare `$` characters) and the Python block has syntax highlighting, KaTeX and Shiki are wired up correctly.

## Inline and display math

Inline math like $E = mc^2$ should appear inline. Display math:

$$
\mathbb{E}_X \left[ \mathrm{KL}(p(y \mid x) \,\|\, q(y \mid x, \theta)) \right]
= \int p(x) \sum_y p(y \mid x) \log \frac{p(y \mid x)}{q(y \mid x, \theta)} \, dx
$$

## Code

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
```

Delete this post when the real content is ready.
