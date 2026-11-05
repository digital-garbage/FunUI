"""Counts-only check: how many shots in your projects describe no real motion.

Read-only. Prints numbers, never prompt text. Not shipped to users.

    python tools/motion_counts.py                      # ~/.funpack_movie/projects
    python tools/motion_counts.py /path/to/projects    # another project folder

A "shot" is a [Shot N] block of a scene's text, or the whole text when it has none. A shot is STATIC when it
has no verb except stative ones (is, stand, sit, look, hold, wear, have...), found with the spaCy tagger
Shot camera already uses (en_core_web_sm). Ratings are the labels the Composer stored: "liked*" and
"disliked*" are split out, any other label (a number, say) is counted under "other rating".
"""
import collections
import glob
import json
import os
import re
import sys

STATIVE = frozenset("""be stand sit look hold wear have lie rest appear seem remain stay face feature show sport dress lean wait
gaze stare hang display contain surround fill""".split())
SHOT = re.compile(r"\[Shot \d+[^\]]*\]")


def bucket(rating):
    r = str(rating or "").strip().lower()
    if not r:
        return "unrated"
    return "disliked" if r.startswith("dislik") else "liked" if r.startswith("lik") else "other rating"


def shots(text):
    parts = [p.strip() for p in SHOT.split(text or "")]
    return [p for p in parts if p]


def main(folder):
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
    except Exception as exc:  # noqa: BLE001 - say what is missing, plainly
        sys.exit(f"needs spaCy and its small English model ({type(exc).__name__}): pip install spacy && python -m spacy download en_core_web_sm")
    files = sorted(glob.glob(os.path.join(os.path.expanduser(folder), "*.json")))
    seen, words = collections.Counter(), collections.defaultdict(list)
    scenes = 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                project = json.load(fh)
        except (OSError, ValueError):
            seen["unreadable project files"] += 1
            continue
        for scene in project.get("scenes", []) if isinstance(project, dict) else []:
            if not isinstance(scene, dict):
                continue
            scenes += 1
            group = bucket(scene.get("rating"))
            rated = scene.get("rated") if isinstance(scene.get("rated"), dict) else {}
            text = (rated.get("text") if group != "unrated" else None) or scene.get("text") or ""   # what the rating is about
            for block in shots(text):
                doc = nlp(block)
                moving = any(t.pos_ == "VERB" and t.lemma_.lower() not in STATIVE for t in doc)
                seen[(group, "moving" if moving else "static")] += 1
                words[group].append(len(block.split()))
    print(f"project files {len(files)}, scenes {scenes}, unreadable {seen.pop('unreadable project files', 0)}")
    print(f"{'group':<14}{'shots':>7}{'static':>8}{'static %':>10}{'avg words':>11}")
    total = [0, 0]
    for group in ("liked", "disliked", "other rating", "unrated"):
        s, m = seen[(group, "static")], seen[(group, "moving")]
        n = s + m
        total[0] += s
        total[1] += n
        w = words[group]
        print(f"{group:<14}{n:>7}{s:>8}{(100 * s / n if n else 0):>9.0f}%{(sum(w) / len(w) if w else 0):>11.1f}")
    print(f"{'all':<14}{total[1]:>7}{total[0]:>8}{(100 * total[0] / total[1] if total[1] else 0):>9.0f}%")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "~/.funpack_movie/projects")
