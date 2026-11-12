# Reproducible environment snapshots

Files in this directory are **release snapshots**, not the primary dependency declarations.

- `requirements/*.txt` describe supported dependency ranges.
- `locks/` records exact environments known to pass a release/course checkpoint.

After creating a clean supported environment and passing `./course.sh test`, capture it with:

```bash
python -m pip freeze --all > locks/cpu-py311.txt
```

A lock must only be committed after CI or a documented reference machine has passed the complete course gate with that exact environment.

Do not hand-edit a lock and call it "known good".
