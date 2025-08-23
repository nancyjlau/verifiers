# arc-agi

> Replace the placeholders below, then remove this callout. Keep the Evaluation Reports section at the bottom intact so reports can auto-render.

### Overview
- **Environment ID**: `arc-agi`
- **Short description**: Abstract reasoning puzzles requiring pattern recognition and grid transformations
- **Tags**: arc-agi, single-turn, reasoning, puzzles

### Datasets
- **Primary dataset(s)**: <name(s) and brief description>
- **Source links**: <links>
- **Split sizes**: <train/eval counts>

### Task
- **Type**: single-turn
- **Parser**: ARCParser (custom grid extraction parser)
- **Rubric overview**: <briefly list reward functions and key metrics>

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval arc-agi
```

Configure model and sampling:

```bash
uv run vf-eval arc-agi   -m gpt-4.1-mini   -n 20 -r 3 -t 1024 -T 0.7   -a '{"key": "value"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments
Document any supported environment arguments and their meaning. Example:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `foo` | str | `"bar"` | What this controls |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |

### Metrics
Summarize key metrics your rubric emits and how they’re interpreted.

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy` | Exact match on target answer |

