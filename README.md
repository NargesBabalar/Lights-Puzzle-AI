# 💡 Lights-Out Puzzle — AI-Based Solver  
_Search Algorithms • Heuristics • Visualization_

> 🎓 **Course**: Artificial Intelligence  
> 🏛️ **University**: University of Tehran  
> 👨‍🏫 **Instructor**: Dr. YaghoobZadeh  

---

## 🧩 Project Overview

This project presents an AI-based solution for the **Lights-Out Puzzle**, a logic-based game played on an `n × n` grid.  
Each cell (light) can be toggled, which also toggles its adjacent neighbors.  
The goal: **turn all lights off** using the least number of moves.

We approach this problem using **classical search algorithms**, comparing their performance in terms of **time**, **space**, and **solution depth**.

---

## 📁 Repository Contents

### 📘 `AI_CA1_Notebook.ipynb`
- Problem modeling: State space, actions, transitions  
- Implementations of:
  - 🔷 Breadth-First Search (BFS)  
  - 🔶 Depth-First Search (DFS)  
  - 🌟 A* Search (with custom heuristics)  
- Grid visualization + interactive solution animation  
- Performance metrics: node expansion, time, memory  

### 📄 `Instruction.pdf`
- Assignment specification  
- Evaluation criteria and required features  

### 📝 `Report.pdf`
- Technical explanation of algorithmic choices  
- Performance comparison (depth, time, space)  
- Heuristic design for A*  
- Strengths, limitations, and future work  

---

## 📊 Key Insights

| Algorithm | Strengths                      | Weaknesses                           |
|-----------|--------------------------------|---------------------------------------|
| **A\***   | Efficient with good heuristic  | Slightly higher setup cost            |
| **BFS**   | Guaranteed optimality          | Very memory-intensive                 |
| **DFS**   | Fast in shallow searches       | Risk of non-termination / deep paths |
