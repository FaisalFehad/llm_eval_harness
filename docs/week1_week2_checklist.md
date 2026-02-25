# Week 1 Checklist

- [ ] Sample 80-100 jobs from DB export into `data/golden_jobs.jsonl`
- [ ] Manually set `label`, `score`, `reasoning` for every row
- [ ] Run `npm run golden:validate:strict`
- [ ] Run `npm run promptfoo:tests`
- [ ] Run `npm run week1:baseline`
- [ ] Save screenshot from Promptfoo matrix

# Week 2 Checklist

- [ ] Inspect failures/disagreements vs labels
- [ ] Update prompt variants (`prompts/scorer_v*.txt`)
- [ ] Run tagged eval for each iteration (`npm run eval:tagged -- --tag ...`)
- [ ] Record quick notes per run
- [ ] Compare trend from run history (`npm run iterations:summary`)
