---
title: Git Tutorial — From First Commit to Research Collaboration
author: Technical Tutorial
language: en
updated: 2026-10-15
---

# Git Tutorial: From First Commit to Research Collaboration

**Ubuntu · GitHub · VS Code · Python / PyTorch / Robot Reinforcement Learning**  
**Edition: October 15, 2026**

This tutorial is for developers who want to version research code, reproduce experiments, and collaborate across workstations, GPU servers, and GitHub. It follows a practical Ubuntu terminal workflow and covers Git fundamentals, installation, commits, remotes, branches, pull requests, diffs, recovery, merge conflicts, ignore rules, advanced commands, a full research workflow, troubleshooting, and VS Code integration.

> **How to use this guide:** Start with Sections 1–5 and work through the end-to-end example in Section 12. Consult the other sections as needed. Replace placeholders such as `YOUR_USERNAME`, `<repo_url>`, and `<commit-id>` before running commands. Repository names, hashes, and sample outputs are illustrative.

## Contents

1. What Git Is: Concepts and the Four Areas
2. Installing and Configuring Git on Ubuntu
3. Creating Your First Local Repository
4. Connecting to GitHub: SSH, Push, Clone, and Sync
5. Git Branch Management
6. Team Collaboration and Pull Requests
7. Inspecting Status, Diffs, and History
8. Undoing Changes and Restoring Older Versions
9. Merge Conflicts and Their Resolution
10. `.gitignore`: Keeping Research Artifacts Out of Git
11. Advanced Commands: Stash, Rebase, Show, and Blame
12. A Complete Research Development Workflow
13. Git Command Cheat Sheet
14. Common Errors and Troubleshooting
15. Using Git in VS Code
16. Best Practices for Research and Reproducibility

---

## 1. What Git Is: Concepts and the Four Areas

**Git is a distributed version control system (VCS).** It records the history of changes to files so that you can compare versions, restore older code, experiment in independent branches, and integrate changes from collaborators.

### 1.1 What problems does Git solve?

- **Version control:** Each commit identifies a traceable state of your code.
- **Safe experimentation:** Try a different reward function or training algorithm without disturbing a stable version.
- **Recovery:** Retrieve accidentally changed code or undo a buggy commit.
- **Collaboration:** Work independently on PPO, SAC, perception, and sim-to-real modules, then integrate them.
- **Cross-machine synchronization:** Use GitHub or GitLab to synchronize source code between a workstation and GPU servers.
- **Reproducibility:** Record the Commit ID used for a training run together with its configuration, seed, and dataset version.

### 1.2 Git versus GitHub, GitLab, and VS Code

| Tool | Purpose |
|---|---|
| **Git** | Local version control, commits, branches, and merging; most operations work offline. |
| **GitHub** | Online Git hosting with pull requests, code review, and automation. |
| **GitLab** | Another platform for hosting Git repositories and collaborating. |
| **VS Code** | A code editor with built-in Source Control support for Git. |

Git is not GitHub. You can run `git init`, `git add`, `git commit`, and `git branch` without any GitHub account or network connection.

### 1.3 Working tree, staging area, local repository, remote repository

```text
                       Local computer
  ┌──────────────────────────────────────────┐
  │ Working Directory / Working tree         │
  │   Files you are currently editing        │
  │                │ git add                 │
  │                ▼                         │
  │ Staging Area / Index                     │
  │   Selected changes for the next commit   │
  │                │ git commit              │
  │                ▼                         │
  │ Local Repository / .git                  │
  │   Commits, branches, and history         │
  └────────────────┬─────────────────────────┘
                   │ git push
                   ▼
             GitHub / Remote repository
                   │ git fetch / git pull
                   ▼
             Local repository / branch
```

The following commands perform **three different operations**:

```bash
git add .
git commit -m "Implement PPO training"
git push
```

- `git add` places **selected file changes** in the staging area.
- `git commit` records the **staged content** in a local commit.
- `git push` sends **local commits** to a remote repository.

A commit is effectively a snapshot of the project at a particular moment, identified by a hash. Uncommitted working-tree edits do **not** get uploaded just because you run `git push`.

### 1.4 Ten essential commands

| Command | Purpose |
|---|---|
| `git init` | Initialize a local repository |
| `git clone URL` | Clone a remote repository |
| `git status` | Inspect the current state |
| `git add PATH` | Stage file changes |
| `git commit -m "..."` | Create a local commit |
| `git push` | Publish commits |
| `git pull` | Fetch and integrate remote changes |
| `git log --oneline` | View compact commit history |
| `git switch -c NAME` | Create and switch to a branch |
| `git merge NAME` | Merge a specified branch |

---

## 2. Installing and Configuring Git on Ubuntu

### 2.1 Install Git

Open an Ubuntu terminal and run:

```bash
sudo apt update
sudo apt install git
git --version
```

You should see a version such as `git version 2.x.y`; the exact version depends on your Ubuntu release and repositories.

### 2.2 Configure the commit author identity

```bash
git config --global user.name "Your Name"
git config --global user.email "your_email@example.com"
```

These values are written into the author identity of new commits. Use an email address associated with GitHub if you want GitHub to attribute the contributions correctly. Inspect the global configuration:

```bash
git config --global --list
```

You can also configure a different identity for **one repository** by omitting `--global`:

```bash
git config user.name "Lab Account"
git config user.email "lab@example.com"
```

### 2.3 Set the initial branch name

```bash
git config --global init.defaultBranch main
```

This affects repositories initialized with `git init` in the future. It does not rename existing branches.

### 2.4 VS Code integration

VS Code includes built-in Git integration. After installing Git, open the project in VS Code and use the Source Control pane. If the `code` command is available on your PATH, run:

```bash
code .
```

For research development, a productive combination is **terminal commands for precise Git operations + VS Code for diffs and conflict resolution**.

---

## 3. Creating Your First Local Repository

Suppose you are building a robot reinforcement learning project:

```text
robot-policy/
├── train.py
├── evaluate.py
├── configs/
│   └── ppo.yaml
├── policies/
│   ├── ppo.py
│   └── sac.py
└── README.md
```

### Step 1: Create the directory and a file

```bash
mkdir robot-policy
cd robot-policy
echo 'print("Hello Git")' > train.py
```

### Step 2: Initialize Git

```bash
git init
```

Git creates a hidden `.git/` directory that stores the repository objects, history, and configuration. Do not casually delete or manually edit this directory.

### Step 3: Check the state

```bash
git status
```

Example output:

```text
On branch main
Untracked files:
  train.py
```

`Untracked` means Git has not started tracking the file.

### Step 4: Stage the changes

```bash
git add train.py
# Or stage all changes under the current directory
git add .
git status
```

`Changes to be committed` indicates staged content. Before using `git add .` in a research project, verify the `.gitignore` rules so that you do not accidentally stage large checkpoints or datasets.

### Step 5: Create a commit

```bash
git commit -m "Initial commit"
```

As your project develops, make one commit per meaningful unit of work:

```bash
git commit -m "Implement PPO training"
git commit -m "Add SAC algorithm"
git commit -m "Fix observation normalization"
```

**A good commit message describes what changed**, rather than simply saying `update` or `fix`.

### Step 6: Read the history

```bash
git log
git log --oneline
```

For example:

```text
f72bc15 Fix PPO reward computation
d43a812 Implement PPO training
a81c32f Initial commit
```

At this point, you can already manage versions **without an internet connection**.

---

## 4. Connecting to GitHub: SSH, Push, Clone, and Sync

### 4.1 Configure an SSH key

SSH authentication is convenient for long-term GitHub development on Ubuntu.

First check for existing keys:

```bash
ls -al ~/.ssh
```

If you do not already have a suitable Ed25519 key, generate one:

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Follow the prompts to choose the key location and passphrase. If a key file already exists, **do not overwrite it without checking**.

Start an SSH agent and load the private key:

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

Display and copy the **public key**:

```bash
cat ~/.ssh/id_ed25519.pub
```

In GitHub, navigate to **Settings → SSH and GPG keys → New SSH key**, paste the public key, and save. The file `id_ed25519` is your **private key**: never publish or upload it. The `.pub` file is the one to share.

Test connectivity:

```bash
ssh -T git@github.com
```

On the first connection, verify GitHub's host fingerprint before accepting it. A successful result generally confirms authentication while noting that shell access is not available.

Reference: [GitHub's official SSH documentation](https://docs.github.com/en/authentication/connecting-to-github-with-ssh).

### 4.2 Push a local project to GitHub

Create an **empty** repository named `robot-policy` on GitHub. If your local repository already has an initial commit, avoid adding a README, `.gitignore`, or License on the website during repository creation; doing so would create a separate initial history.

Then, locally:

```bash
cd robot-policy
git remote add origin git@github.com:YOUR_USERNAME/robot-policy.git
git remote -v
git branch -M main
git push -u origin main
```

Interpretation:

- `origin` is the conventional **name** of a remote; it is not a special server.
- `git remote -v` shows the fetch and push URLs.
- `git branch -M main` names the current branch `main`.
- `-u` establishes upstream tracking so later pushes usually need only `git push`.

### 4.3 Clone an open-source project

For example:

```bash
git clone https://github.com/ajwagen/dsrl.git
cd dsrl
```

With SSH configured and appropriate access, you can instead use the repository's SSH URL:

```bash
git clone git@github.com:ajwagen/dsrl.git
```

`clone` downloads the repository history, creates the project directory, and normally configures an `origin` remote. Do **not** run `git init` again inside a cloned repository.

| HTTPS | SSH |
|---|---|
| Public repositories can typically be cloned anonymously | Typically requires SSH authentication |
| Uses an `https://...` URL | Uses a `git@github.com:...` URL |
| Pushing needs appropriate account credentials | Key-based authentication makes pushing convenient |

### 4.4 Update local code: Fetch versus Pull

```bash
cd dsrl
git fetch origin
```

`fetch` retrieves remote objects and updates remote-tracking references. It **does not automatically merge them into the current local branch**.

If your local `main` tracks `origin/main`, you may then run:

```bash
git merge origin/main
```

A common shorthand is:

```bash
git pull
```

`pull` performs `fetch` followed by integration, using merge or rebase according to the command options and Git configuration. To avoid an unexpected merge commit when updating a stable branch, beginners may prefer:

```bash
git pull --ff-only
```

This succeeds only when the update can be fast-forwarded; if the histories have diverged, Git stops and lets you decide how to integrate them.

> **Note:** You normally cannot push directly to another team's open-source repository unless you have permission. You can Fork the repository first and work in your own Fork, or contribute through the project's pull request process.

---

## 5. Git Branch Management

Branches let you maintain a stable implementation and experimental features side by side. For example, `main` can keep stable PPO code while `feature/sac` contains a new SAC implementation:

```text
A --- B ---------------- M       main
       \                /
        C --- D -------/         feature/sac
```

Here `M` is a possible merge commit. If the merge can be fast-forwarded, Git may not create a separate merge commit.

### 5.1 List, create, and switch branches

```bash
git branch                         # List local branches
git branch feature/sac             # Create only
git switch -c feature/sac          # Create and switch (alternative)
git switch main                    # Switch to main
git switch feature/sac             # Return to the feature branch
```

If you already created the branch using `git branch feature/sac`, switch with `git switch feature/sac`; do not run `switch -c` with the same name again.

Switching branches updates files in your working tree. Git may block the switch if an uncommitted modification would be overwritten. Commit the work or use `git stash` first.

### 5.2 Develop and commit on a feature branch

```bash
git switch -c feature/sac
# Edit policies/sac.py
git add policies/sac.py
git commit -m "Implement SAC policy"
```

This advances `feature/sac`; it does not directly change `main`.

### 5.3 Merge into the main branch

```bash
git switch main
git merge feature/sac
```

If there are no conflicts, Git completes the merge. Otherwise, see Section 9.

### 5.4 Delete a finished branch

```bash
git branch -d feature/sac
```

The `-d` option checks that the branch has been safely integrated. To delete the corresponding remote branch:

```bash
git push origin --delete feature/sac
```

For shared repositories, it is generally better to merge feature branches via a reviewed GitHub Pull Request into a protected `main` branch.

---

## 6. Team Collaboration and Pull Requests

A research group might organize work as follows:

```text
main
├── feature/ppo
├── feature/sac
├── feature/pose-estimation
└── feature/sim2real
```

A recommended workflow is **Feature Branch → Push → Pull Request → Review → Merge**.

### 6.1 Start from the current main branch

```bash
git switch main
git pull --ff-only
git switch -c feature/pose-estimation
```

### 6.2 Implement, inspect, and commit

```bash
# Modify the pose estimation module
git status
git diff
git add .
git commit -m "Implement visual pose estimation"
```

### 6.3 Push the branch and open a PR

```bash
git push -u origin feature/pose-estimation
```

Create a Pull Request on GitHub, typically from `feature/pose-estimation` into `main`. Other team members can inspect changes, review code, discuss implementation details, run CI checks, and approve the integration.

A useful team rule: **do not make large, untested algorithm changes directly on a shared stable branch**. Validate them in isolation first.

---

## 7. Inspecting Status, Diffs, and History

When debugging a training run, knowing what the code **actually changed** is often more valuable than remembering what you *intended* to change.

### 7.1 Inspect the working tree

```bash
git status
```

The output identifies your current branch, staged edits, unstaged edits, and untracked files.

### 7.2 Compare code

```bash
git diff                     # Working tree vs staging area
git diff --staged            # Staging area vs latest commit
git diff -- train.py         # Changes in one file
```

A diff might look like:

```diff
- learning_rate = 3e-4
+ learning_rate = 1e-4
```

The `-` line is removed; the `+` line is added.

### 7.3 Visualize the commit graph

```bash
git log --oneline
git log --oneline --graph --all
```

For example:

```text
* a34bd21 (HEAD -> main) Merge SAC implementation
|\
| * d72f8a1 Implement SAC algorithm
| * f61b2c9 Add replay buffer
|/
* b83fc92 Implement PPO
* 28ad613 Initial commit
```

Make a habit of combining `git status`, `git diff`, and `git log --oneline --graph --all`. Before each commit, confirm that you have not staged temporary artifacts, model weights, or secrets.

---

## 8. Undoing Changes and Restoring Older Versions

Recovery depends on whether changes are **unstaged, staged, committed, or already shared**. The following commands are not interchangeable.

### 8.1 Unstaged changes: Discard one file's working-tree edits

```bash
git restore train.py
```

This restores `train.py` in the working tree from the staging area. **Unbacked working-tree edits may be lost.**

### 8.2 Staged changes but no commit: Unstage

```bash
git restore --staged train.py
```

The edits remain in the working tree, but they will not be included in the next commit unless staged again.

### 8.3 A local, unshared commit: Undo the commit but keep edits

Assume the latest commits are:

```text
a82bc31 Bad reward modification
b41d2ab Working PPO version
```

Undo the last commit while retaining its staged changes:

```bash
git reset --soft HEAD~1
```

Undo the last commit and unstage the changes, but retain the working-tree edits:

```bash
git reset HEAD~1
```

Undo the last commit and discard the corresponding working-tree changes:

```bash
git reset --hard HEAD~1
```

**`reset --hard` is destructive.** Verify that you do not need the affected code, experiment results, or uncommitted work before using it.

### 8.4 A pushed/shared commit: Use Revert

```bash
git revert a82bc31
```

`revert` creates a new commit that attempts to cancel the changes introduced by an earlier commit, preserving an auditable shared history. Conflicts are possible if later commits have modified the same code.

| Operation | Effect | Typical use |
|---|---|---|
| `git restore FILE` | Discard unstaged file edits | Abandon temporary local changes |
| `git restore --staged FILE` | Unstage, keep edits | Accidental staging |
| `git reset --soft HEAD~1` | Move branch pointer, keep staging | Refine unshared commits |
| `git reset HEAD~1` | Move pointer, keep working edits | Restage selected changes |
| `git reset --hard HEAD~1` | Move pointer and discard edits | Only when no recovery is needed |
| `git revert HASH` | Add a reverse commit | Undo an already shared commit |

General principle: **use `reset` cautiously for private history; prefer `revert` for shared history.**

---

## 9. Merge Conflicts and Their Resolution

A merge conflict may arise when two branches modify overlapping lines and Git cannot choose an automatic resolution.

Suppose two branches changed `train.py`:

```text
<<<<<<< HEAD
learning_rate = 0.0001
=======
learning_rate = 0.0003
>>>>>>> feature/sac
```

- Between `<<<<<<< HEAD` and `=======` is the current branch's version.
- Between `=======` and `>>>>>>> feature/sac` is the incoming branch's version.

### 9.1 Resolve a conflict

1. Run `git status` to identify conflicting files.
2. Open the file and understand the intent of both changes.
3. Keep one side, or write a result that correctly combines both behaviors.
4. **Remove all conflict markers** (`<<<<<<<`, `=======`, `>>>>>>>`).
5. Run tests and validate the behavior.
6. Stage the resolved file and complete the merge commit.

For example, choose:

```python
learning_rate = 0.0001
```

Then run:

```bash
git add train.py
git commit
```

To cancel an in-progress merge:

```bash
git merge --abort
```

For conflicts during a rebase, use `git rebase --continue` or `git rebase --abort` instead. `git merge --abort` is for a merge operation.

### 9.2 Resolve conflicts using VS Code

VS Code's Merge Editor compares the current version, the incoming version, and the merged result. Regardless of whether you use the editor or terminal, run tests afterward—especially for reward functions, action scaling, robot reference frames, and other changes whose semantics are not obvious from a text diff.

---

## 10. `.gitignore`: Keeping Research Artifacts Out of Git

Machine learning projects routinely produce gigabytes of datasets, checkpoints, videos, and logs. Such files are usually **not suitable for an ordinary source-code Git repository**.

### 10.1 A useful template

Create `.gitignore` in the project root:

```bash
touch .gitignore
nano .gitignore
```

Example:

```gitignore
# Python cache
__pycache__/
*.pyc

# Local environments
.venv/
venv/

# Secrets
.env
*.pem

# Model artifacts
*.pt
*.pth
*.ckpt
checkpoints/

# Training outputs
outputs/
logs/
wandb/

# Large datasets
data/
datasets/

# ROS recordings
*.bag
*.db3

# Optional IDE state
.vscode/
```

**Adapt these rules to your project.** Some projects intentionally version tiny test weights or share certain `.vscode/` settings, so blanket exclusions may not be appropriate.

### 10.2 What `.gitignore` does—and does not—do

Ignore rules mostly affect **untracked** files. If `checkpoint.pt` is already tracked, adding `*.pt` to `.gitignore` does not automatically stop tracking it.

To remove a file from Git tracking while keeping the local copy:

```bash
git rm --cached checkpoint.pt
git commit -m "Stop tracking checkpoint"
```

This does not remove old copies from existing commits. If you accidentally published an API key, password, or other secret, revoke/rotate the affected credentials promptly and evaluate whether public history must be cleaned.

### 10.3 Separate source code, data, and models

Recommended division:

- Git: source code, configurations, scripts, documentation, and small test assets.
- Object storage or experiment tracking: checkpoints, videos, logs, and large datasets.
- Experiment metadata: Commit ID, dataset version, weight checksum, and training parameters.

For versioned large artifacts, separately consider Git LFS, DVC, or a specialized model/data management solution.

---

## 11. Advanced Commands: Stash, Rebase, Show, and Blame

### 11.1 `git stash`: Temporarily shelve uncommitted work

Suppose you are implementing SAC but must switch to `main` to fix an urgent issue:

```bash
git stash
```

By default, this mainly saves changes to tracked files. Include untracked files by running:

```bash
git stash push -u -m "WIP SAC training"
```

Then:

```bash
git switch main
# Fix the issue and commit it
git switch feature/sac
git stash pop
```

| Command | Purpose |
|---|---|
| `git stash list` | List saved stashes |
| `git stash show -p` | Inspect the most recent stash's patch |
| `git stash pop` | Restore a stash and remove it if successful |
| `git stash apply` | Restore a stash but keep the stash record |

Stash application can also conflict; it is not a substitute for reliable long-term backups.

### 11.2 `git rebase`: Move experimental work onto a newer base

Original history:

```text
main:    A -- B -- C
feature: A -- B -- D -- E
```

Run:

```bash
git switch feature
git rebase main
```

Conceptually, the result is:

```text
main:    A -- B -- C
feature: A -- B -- C -- D' -- E'
```

`rebase` replays changes and creates new commit IDs. It is useful for organizing **your own private branches**, but do not casually rewrite shared history that other developers are using.

### 11.3 Inspect a commit or trace a changed line

```bash
git show <commit-id>       # Full details of one commit
git blame train.py         # Last modifying commit for each line
```

`git blame` can help answer “when was reward scaling changed?” or “which commit changed the control frequency?” It is a historical diagnostic tool, not a judgment about individual responsibility.

### 11.4 Inspect local and remote branches

```bash
git branch -a
```

Example:

```text
* main
  feature/sac
  remotes/origin/main
  remotes/origin/feature/ppo
```

Remote-tracking references (`remotes/origin/...`) represent Git's locally recorded state of remote branches. Use `git fetch` to refresh them.

---

## 12. A Complete Research Development Workflow

Goal: Add `residual_velocity.py` to a reinforcement learning codebase, implement Advantage-Guided Velocity Loss, test the method, and integrate it.

### Phase 1: Clone your repository

```bash
git clone git@github.com:YOUR_USERNAME/robot-policy.git
cd robot-policy
```

If you start from someone else's open-source project, Fork it first or follow its documented contribution process.

### Phase 2: Branch from the latest `main`

```bash
git switch main
git pull --ff-only
git switch -c experiment/residual-velocity
```

### Phase 3: Develop and inspect changes

Your project could look like:

```text
robot-policy/
├── policies/
│   ├── sac.py
│   ├── ppo.py
│   └── residual_velocity.py
├── configs/
│   └── residual_velocity.yaml
└── train.py
```

After editing:

```bash
git status
git diff
```

### Phase 4: Commit the first self-contained feature

```bash
git add policies/residual_velocity.py
git add configs/residual_velocity.yaml
git add train.py
git commit -m "Implement residual velocity adaptation"
```

### Phase 5: Add the next algorithmic component

```bash
# Implement and test Advantage-Guided Loss
git add .
git commit -m "Add advantage-guided velocity loss"
```

This produces a clear branch history:

```text
A --- B --- C            experiment/residual-velocity
      │     │
      │     └─ Add advantage-guided velocity loss
      └─────── Implement residual velocity adaptation
```

### Phase 6: Publish your branch

```bash
git push -u origin experiment/residual-velocity
```

For later updates, typically:

```bash
git push
```

### Phase 7: Validate and merge

For a personal repository, you may use:

```bash
git switch main
git pull --ff-only
git merge experiment/residual-velocity
git push origin main
```

For a shared team repository, open a GitHub Pull Request and merge after review, tests, and permission checks. **A merged branch is not automatically a reproducible experiment**: still record dependency versions, dataset versions, random seeds, and the checkpoint's exact Commit ID.

---

## 13. Git Command Cheat Sheet

| Category | Command | Purpose |
|---|---|---|
| Setup | `git init` | Create a repository |
| Clone | `git clone URL` | Download a repository |
| Status | `git status` | Inspect current state |
| Diff | `git diff` | Inspect unstaged changes |
| History | `git log --oneline` | Browse commit history |
| Stage | `git add .` | Stage changes in the current directory |
| Commit | `git commit -m "msg"` | Create a commit |
| Publish | `git push` | Send local commits to a remote |
| Fetch | `git fetch` | Retrieve remote state without merging |
| Sync | `git pull --ff-only` | Fast-forward-only update |
| Branch | `git branch` | List local branches |
| Branch | `git switch -c name` | Create and switch branch |
| Branch | `git switch name` | Switch to an existing branch |
| Merge | `git merge name` | Merge a named branch |
| Undo | `git restore file` | Discard unstaged file edits |
| Undo | `git restore --staged file` | Unstage a file |
| Undo | `git revert HASH` | Add a reversing commit |
| Shelf | `git stash` | Temporarily save edits |
| Shelf | `git stash pop` | Restore a stash |
| Inspect | `git show HASH` | Examine a commit |
| Inspect | `git blame file` | Inspect per-line commit attribution |

---

## 14. Common Errors and Troubleshooting

### 14.1 `fatal: not a git repository`

Git cannot find a valid repository in the current directory or its parents. Check:

```bash
pwd
ls -a
```

Make sure you are in the project directory. Run `git init` only if you actually intend to create a **new** repository.

### 14.2 `Permission denied (publickey)`

SSH authentication failed. Check:

```bash
ssh -T git@github.com
```

Verify that the SSH agent has your correct private key loaded and that its corresponding public key is attached to the intended GitHub account.

### 14.3 `Updates were rejected`

This usually means that the remote has commits missing from your local history, so the push cannot fast-forward. Inspect first:

```bash
git fetch origin
git status
```

For your own feature branch with a correctly configured upstream, you can choose this workflow if it matches your team's conventions:

```bash
git pull --rebase
# Resolve potential conflicts and run tests
git push
```

**Do not blindly use `git push --force`** just to make the error disappear; it can overwrite collaborators' commits.

### 14.4 `Your local changes would be overwritten`

Uncommitted local changes might be overwritten by a switch or update. Choose one of the following before retrying:

```bash
# Option A: Commit the current work
git add .
git commit -m "Save work in progress"
```

```bash
# Option B: Temporarily stash it, including untracked files
git stash push -u
```

### 14.5 A general troubleshooting rule

When Git reports an unfamiliar error, start with `git status`, read its suggested next steps, and determine whether you are in a merge, rebase, or ordinary working-tree state. Avoid chains of `reset --hard`, `clean -fd`, or force-push commands when you do not understand the current state.

---

## 15. Using Git in VS Code

Open the repository with:

```bash
code .
```

Open the **Source Control** pane, commonly with `Ctrl + Shift + G`.

| Action | VS Code workflow |
|---|---|
| View changed files | Source Control file list |
| Compare versions | Click a modified file in the list |
| Stage edits | `+` / Stage Changes |
| Commit | Enter a message and choose Commit |
| Push | Push or Sync Changes |
| Switch branches | Status bar branch label or Command Palette |
| Resolve conflicts | Merge Editor |

For example, change:

```python
reward = success_reward
```

to:

```python
reward = success_reward + progress_reward
```

VS Code's Diff Editor clearly shows the removed and added lines. For RL experiments with many configuration files, reviewing the complete diff before each commit often reveals accidental hyperparameter changes.

**Recommended combination:** Use the terminal for reproducible commands and precise control, and use VS Code to review large diffs and resolve conflicts.

---

## 16. Best Practices for Research and Reproducibility

### 16.1 Name branches by the research task

| Research scenario | Suggested branch |
|---|---|
| Implement PPO or SAC | `feature/ppo` / `feature/sac` |
| Explore a different reward | `experiment/reward-v2` |
| Adapt an online residual velocity field | `experiment/residual-velocity` |
| Update pose estimation or camera calibration | `feature/pose-estimation` |
| Fix deployed robot control latency | `fix/control-latency` |
| Reproduce a paper's experiment | Pin the code using a Commit ID or a Tag |

### 16.2 Record the Commit ID with every training run

```bash
git rev-parse HEAD
```

Save it alongside the experiment configuration, for example:

```yaml
experiment:
  name: residual_velocity_v3
  git_commit: a82bc31
  random_seed: 42
training:
  learning_rate: 0.0003
  batch_size: 256
```

The short hash above is illustrative; for production experiment records, store the **full hash** returned by the command.

> Important: A Commit ID identifies committed code only. If you launch an experiment with uncommitted modifications, the hash alone cannot fully reproduce that run. Check `git status` before official runs, and record a patch or create a dedicated experiment commit if needed.

### 16.3 Separate code, configuration, and large files

- **Git repository:** Source code, configs, entry points, evaluation scripts, README, environment documentation.
- **Experiment output storage:** TensorBoard/W&B logs, videos, weights, and large caches—usually ignored by Git.
- **Reproducibility metadata:** Commit ID, environment/dependency versions, dataset version, seed, simulator version, and GPU/driver information.
- **Robot deployment:** Tag a verified policy release (e.g., `v1.0-deploy`) and store compatible controller settings and hardware versions.

Example: tag a verified deployment version:

```bash
git tag -a v1.0-deploy -m "Validated robot deployment"
git push origin v1.0-deploy
```

### 16.4 A beginner's complete workflow

```bash
# 1. Clone the project
git clone <repo_url>
cd <repo_name>

# 2. Check status
git status

# 3. Create an experiment branch
git switch -c experiment/new-method

# 4. Edit and inspect changes
git diff

# 5. Commit
git add .
git commit -m "Implement new method"

# 6. First push of this branch
git push -u origin experiment/new-method

# 7. Subsequent changes and pushes
git add .
git commit -m "Improve training"
git push
```

Knowing this workflow is sufficient for most day-to-day solo research development. Build on it by learning conflict resolution, rebasing, pull requests, and your team's review conventions.

### 16.5 Official references

- [Pro Git (Chinese)](https://git-scm.com/book/zh/v2)
- [Pro Git (English)](https://git-scm.com/book/en/v2)
- [Git Command Reference](https://git-scm.com/docs)
- [GitHub: Connect with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
- [GitHub: Pull Request Documentation](https://docs.github.com/en/pull-requests)

---

**Conclusion:** Git's main contribution to research is not simply “uploading code to GitHub.” It is connecting **source-code versions, experimental changes, and reproducible results**. Start with small commits, separate experiment branches, review diffs, and keep large artifacts out of the main source repository. Add team collaboration practices as your projects grow.
