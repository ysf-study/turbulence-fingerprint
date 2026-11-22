# Publishing this repository

Two things have to happen before the link can go into the manuscript: the code has
to be on GitHub, and it needs a DOI (Neurocomputing applies Elsevier's research
data policy Option C, which requires a repository link at submission).

## 0. Repository settings

Set these once, in **Settings → General** and the **About** panel, so that the
repository looks like software rather than a folder dump:

| Field | Value |
|---|---|
| Description | `Code and derived data for "Higher-Order Intermittency in Learned Neural Motion Representations" (Neurocomputing submission)` |
| Topics | `ai-generated-video-detection`, `optical-flow`, `turbulence`, `reproducibility`, `xgboost`, `interpretable-ml` |
| Website | the Zenodo DOI URL once minted |
| Features | Issues on; Wiki off; Discussions optional |
| Default branch | `main` |

GitHub's *Community Standards* checklist should read 3/3 once
`README.md`, `LICENSE`, `CONTRIBUTING.md` and `CODE_OF_CONDUCT`-equivalent
(`AUTHORS.md` here) are in place; `NOTICE.md` and `.zenodo.json` cover the
provenance and deposit metadata that checklist does not ask about.

## 1. Push to GitHub

Create an empty repository named `turbulence-fingerprint` (no README, no licence —
this folder already has them), then:

```bash
cd turbulence-fingerprint
git init -b main
git add .
git commit -m "Code and derived data for the TF manuscript (v1.1.0)"

# replace ysf-study with your account or organisation
git remote add origin https://github.com/ysf-study/turbulence-fingerprint.git
git push -u origin main
```

Check before committing that `git status` shows no `*.npz`, `*.mp4`, `*.pth`,
`__pycache__` or `results/fusion.csv` — `.gitignore` covers them, but the data
folder is the one place where a stray file would be a licence problem.

The CI workflow runs on the first push: `verify_release.py`, `pytest`, a full
re-fit diffed against `results/reference/`, `ruff`, and a metadata parse. A red
badge on a submission repository is worse than no badge, so fix it before sending
the link to the editor.

Replace `ysf-study` in four places or the badge, the clone URL and the container
labels stay broken: `README.md` (badge URLs), `CITATION.cff`,
`pyproject.toml`, `Dockerfile`.

## 2. Get a DOI

GitHub alone is not a data repository: a repository can be edited or deleted, so it
has no persistent identifier. Archive a tagged release to Zenodo:

1. Sign in to <https://zenodo.org> with GitHub.
2. Settings → GitHub → flip the switch for `turbulence-fingerprint`.
3. On GitHub, create a release: tag `v1.1.0`, title "Manuscript submission version".
4. Zenodo mints a DOI and shows it next to the release.

The DOI looks like `10.5281/zenodo.1234567`; the citable URL is
`https://doi.org/10.5281/zenodo.1234567`.

`.zenodo.json` controls the deposit record: title, description, creators,
keywords and licence. Zenodo reads it at release time and prefers it over the
GitHub metadata, so edit that file rather than the web form.

Suggested release notes:

```
Manuscript submission version.

- 1,635-clip four-dimensional feature table and the fixed five-fold split
- per-clip scores for the foundation-model baseline and the D3 detector
- reference outputs of the published run, byte-reproducible on pinned versions
- 26 integrity checks and 15 tests, no GPU required
```

## 3. Fill the DOI into three places

| Place | What to change |
|---|---|
| `README.md` | the `DOI` badge and its link |
| `CITATION.cff` | `doi:` and the `url` / `repository-code` placeholders |
| `.zenodo.json` | nothing on the first release; on later ones bump `version` |
| manuscript | `8_appendix.tex` (Data availability), `7_conclusion.tex`, Cover Letter — all three must carry the **same** string |

## 4. Keep the review version frozen

Tag `v1.1.0` at submission and do not push to `main` while the paper is under
review. If you must fix a typo, tag `v1.1.1` and note in the response letter that
the review version remains `v1.1.0`. A repository whose content changes under a
reviewer is the fastest way to turn a reproducibility claim into a complaint.

## 5. After acceptance

- Add the paper's DOI to `.zenodo.json` under `related_identifiers`
  (`relation: "isSupplementTo"`, `scheme: "doi"`) and to `CITATION.cff`.
- Update the `preferred-citation` block in `CITATION.cff` with the volume, issue
  and page range once they exist.
- Re-release as `v1.2.0` if the repository changed during revision; otherwise
  leave the reviewed tag alone and cite it.

