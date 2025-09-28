---
layout: post
title: "GitHub 협업 표준 워크플로우"
date: 2025-06-30
categories: [Programming, GitHub]
tags: [github, git, collaboration, workflow, pull request, fork, code review]
---

## GitHub 협업 표준 워크플로우 요약

### 명령어

| 단계               | 명령어                                                                                                                                   |
| :----------------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| 1.원격 설정        | GitHub에서 `Fork` → 내 repo(origin) 생성                                                                                                 |
| 2.로컬 clone       | `git clone {my-repo-address}`                                                                                                            |
| 3.원격 설정        | `git remote add upstream {team-repo-address}`                                                                                            |
| 4.로컬 브랜치 생성 | `git branch {branch-name}` & `git switch {branch-name}`                                                                                  |
| 5.작업 & 커밋      | `git add {file-name}` & `git commit`                                                                                                     |
| (6.로컬 병합)      | `git switch main` → `git merge {branch-name}`                                                                                            |
| 7.push             | `git push origin main` <br> (6.로컬 병합 안했다면 최초에는 `git push -u origin {branch-name}` <br> 이후 `git push origin {branch-name}`) |
| 8.PR생성           | GitHub 웹에서 `Compare & pull request` 클릭 <br> `base: {받을 upstream branch} ← compare: {보낼 origin branch}`                          |

---

### 기타

* PR생성 후 code review결과 추가 작업사항은 main 말고 열려있는 branch에서 작업 후 commit (위의 4,5,6,7 중 필요한 단계 반복)
* Fork 없이 협업 repo를 그대로 내 local에 clone한 경우
  * 별도 upstream은 없음 (origin 이 됨)
  * 이 때는 반드시 branch를 push할 것 (6.로컬 병합 생략)
  * 권한 없으면 push 안됨
* upstream branch를 로컬로 업데이트할 때 명령어
  * git fetch upstream main
  * git merge upsteram/main
  * 주의사항: 이 때 내 local에 충돌하는 파일이 commit되어있지 않다면 merge 안됨

---

### 참고

* **항상 fork 후 작업하고 PR을 보내는 방식이 안전함**
* fork없이 직접 clone 방식은 **권한자만 사용** (push 권한이 없으면 PR도 불가)
* 브랜치를 따로 만들어서(`feature/...` 등) 작업 후 PR 보내기
* PR을 보낸다고 해서 upstream에 해당 브랜치가 **자동 생성되지는 않음**
