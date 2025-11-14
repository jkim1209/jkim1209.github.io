---
layout: post
title: "GitHub Collaboration Standard Workflow"
date: 2025-06-30
categories: [Programming, GitHub]
tags: [github, git, collaboration, workflow, pull request, fork, code review]
---

## GitHub Collaboration Standard Workflow Summary

### Commands

| Step                  | Command                                                                                                                                                            |
| :-------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.Remote Setup        | `Fork` on GitHub → Create my repo(origin)                                                                                                                          |
| 2.Local Clone         | `git clone {my-repo-address}`                                                                                                                                      |
| 3.Remote Setup        | `git remote add upstream {team-repo-address}`                                                                                                                      |
| 4.Create Local Branch | `git branch {branch-name}` & `git switch {branch-name}`                                                                                                            |
| 5.Work & Commit       | `git add {file-name}` & `git commit`                                                                                                                               |
| (6.Local Merge)       | `git switch main` → `git merge {branch-name}`                                                                                                                      |
| 7.Push                | `git push origin main` <br> (If 6.Local Merge not done, initially `git push -u origin {branch-name}` <br> then `git push origin {branch-name}`)                   |
| 8.Create PR           | Click `Compare & pull request` on GitHub web <br> `base: {receiving upstream branch} ← compare: {sending origin branch}`                                           |

---

### Additional Notes

* After creating PR, for additional work based on code review results, work on the open branch (not main) and commit (repeat necessary steps from 4,5,6,7 above)
* When collaborating repo is cloned directly to local without Fork
  * No separate upstream (origin becomes it)
  * In this case, must push the branch (skip 6.Local Merge)
  * Cannot push without permission
* Commands to update upstream branch to local
  * git fetch upstream main
  * git merge upstream/main
  * Caution: If conflicting files in local are not committed, merge will not work

---

### References

* **Always working after fork and sending PR is the safe approach**
* Direct clone without fork should be used **only by authorized users** (PR also impossible without push permission)
* Create a separate branch (`feature/...` etc.) to work on and send PR
* Sending PR does **not automatically create** that branch in upstream
