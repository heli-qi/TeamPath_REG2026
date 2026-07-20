#!/usr/bin/env python3
"""Faithful per-organ pathology report generator. Given the CoT answers for a case,
reconstruct the final report to (near-)exactly match the ground-truth format.
Header '{organ}, {procedure};' + numbered diagnosis list + organ-specific detail."""


def _lc1(s):
    return s[:1].lower() + s[1:] if s else s


def _prostate_suffix(ans):
    gs = ans.get("What is the Gleason score?", "").strip()
    gg = ans.get("What is the grade group?", "").strip()
    g4p = ans.get("Is there any Gleason pattern 4 present?", "").strip().lower()
    g4 = ans.get("What is the percentage of Gleason pattern 4?", "").strip()
    vol = ans.get("What is the tumor volume?", "").strip()
    s = ""
    if gs:
        s += f", Gleason's score {gs}"
    if gg:
        s += f", {_lc1(gg)}"
    if g4p.startswith("yes") and g4:
        s += f" (Gleason pattern 4: {g4})"
    if vol:
        s += f", tumor volume: {vol}"
    return s


def _dcis_block(ans, ind):
    t = ans.get("What is the architectural pattern of lesion?")
    ng = ans.get("What is the nuclear grade of lesion?")
    necp = ans.get("Is there any necrosis present?", "").strip().lower()
    nect = ans.get("What is the type of necrosis?", "").strip()
    parts = []
    if t:
        parts.append(f"\n{ind}- Type: {t}")
    if ng:
        parts.append(f"\n{ind}- Nuclear grade: {ng}")
    if necp.startswith("yes"):
        nt = nect.replace(" necrosis", "").strip()
        parts.append(f"\n{ind}- Necrosis: Present ({nt})" if nt else f"\n{ind}- Necrosis: Present")
    elif necp.startswith("no"):
        parts.append(f"\n{ind}- Necrosis: Absent")
    return "".join(parts)


# stomach gastritis sub-findings (ordered)
_GASTRITIS_SUBS = [
    ("Is there any intestinal metaplasia present?", "intestinal metaplasia"),
    ("Is there any foveolar epithelial hyperplasia present?", "foveolar epithelial hyperplasia"),
    ("Is there any lymphoid aggregate present?", "lymphoid aggregate"),
    ("Is there any lymphoid follicle present?", "lymphoid follicle"),
]


def _with_list(subs):
    """Render '(with) X' for 1 sub, or numbered 'with 1) .. \\n 2) ..' for 2+."""
    if not subs:
        return ""
    if len(subs) == 1:
        return f" with {subs[0]}"
    out = f"\n  with 1) {subs[0]}"
    for i, s in enumerate(subs[1:], 2):
        out += f"\n       {i}) {s}"
    return out


def _diag_detail(organ, d, ans, ind):
    dl = d.lower()
    o = organ.lower()
    if "invasive carcinoma of no special type" in dl or "invasive breast carcinoma" in dl:
        tf = ans.get("What is the score for tubular differentiation?")
        ng = ans.get("What is the score for nuclear pleomorphism?")
        mi = ans.get("What is the score for mitotic rate?")
        if tf and ng and mi:
            return f" (Tubule formation: {tf}, Nuclear grade: {ng}, Mitoses: {mi})"
    if "ductal carcinoma in situ" in dl:
        return _dcis_block(ans, ind)
    if "prostate" in o and "adenocarcinoma" in dl:
        return _prostate_suffix(ans)
    if "adenoma" in dl and ("colon" in o or "rectum" in o or "stomach" in o):
        gd = ans.get("What is the grade of dysplasia?", "").strip()
        if gd:
            return f" with {_lc1(gd)} dysplasia"
    if "stomach" in o and "gastritis" in dl:
        subs = [name for q, name in _GASTRITIS_SUBS if ans.get(q, "").strip().lower().startswith("yes")]
        return _with_list(subs)
    return ""


def gen_report(ans, level=2):
    organ = ans.get("What is the organ?", "").strip()
    proc = _lc1(ans.get("What is the procedure?", "").strip())
    diags = []
    for i in range(1, 5):
        d = ans.get(f"What is the #{i} diagnosis?")
        if d and d.strip():
            diags.append(d.strip())
    if not diags:
        d = ans.get("What is the #1 diagnosis?") or ans.get("What is the histologic type of neoplasm?")
        if d and d.strip():
            diags = [d.strip()]
    header = f"{organ}, {proc};"
    numbered = len(diags) > 1
    ind = "     " if numbered else "  "       # detail indent: 5 spaces if numbered else 2
    lines = []
    for i, d in enumerate(diags, 1):
        d2 = d.replace("Micro-invasive", "Microinvasive")   # report drops the hyphen
        tail = ""
        if " with glandular involvement" in d2:                # cervix
            d2 = d2.replace(" with glandular involvement", "")
            tail = "\n  with glandular involvement"
        elif " with involvement of" in d2:                     # bladder invasive ca
            j = d2.index(" with involvement of")
            tail = ",\n  with involvement of" + d2[j + len(" with involvement of"):]
            d2 = d2[:j]
        prefix = f"{i}. " if numbered else ""
        detail = _diag_detail(organ, d2, ans, ind) if level >= 2 else ""
        lines.append(f"  {prefix}{d2}{detail}{tail}")
    body = "\n".join(lines)
    note = ""
    if organ.lower().startswith("urinary bladder"):
        m = ans.get("Is there any muscularis propria present?", "").strip().lower()
        # separator: single \n only when multi-dx AND a "No tumor present" item; else \n\n
        notumor = any("no tumor present" in d.lower() for d in diags)
        sep = "\n" if (numbered and notumor) else "\n\n"
        if m.startswith("yes"):
            note = sep + "Note) The specimen includes muscle proper."
        elif m.startswith("no"):
            note = sep + "Note) The specimen does not include muscle proper."
    return header + "\n" + body + note


if __name__ == "__main__":
    import json, sys, collections
    BASE = "/workspace/sota_run"
    data = json.load(open(f"{BASE}/train_CoT.json", encoding="utf-8"))
    CK = "chain-of-thought"

    def answers(case):
        d = {}
        for s in case.get(CK, []):
            q = (s.get("question") or "").strip()
            if q not in d:
                d[q] = (s.get("answer") or "").strip()
        return d

    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    byorg = collections.Counter(); byorg_ex = collections.Counter(); exact = n = 0
    for case in data:
        a = answers(case)
        gt = a.get("What is the final pathology report?")
        if not gt:
            continue
        n += 1; o = case.get("organ"); byorg[o] += 1
        if gen_report(a, level=lvl) == gt.replace("\\n", "\n"):
            exact += 1; byorg_ex[o] += 1
    print(f"OVERALL exact report match: {exact}/{n} ({100*exact/n:.1f}%)")
    for o in byorg:
        print(f"  {o:10} {byorg_ex[o]:5}/{byorg[o]:5}  ({100*byorg_ex[o]/byorg[o]:.0f}%)")
