# CS343/465 Graph Data Science — Final Project
### Graph Machine Learning on the POLE Crime Dataset
**Habib University | Spring 2026**

This repository contains the complete implementation and report for our 
CS343/465 Graph Data Science final project. We apply three Graph Machine 
Learning techniques — Community Detection, Node Classification, and 
Link Prediction — to the Manchester POLE (Person, Object, Location, 
Event) crime dataset using the Neo4j Graph Data Science (GDS) library.

---

---

## Dataset

The dataset is a synthetic yet realistic instantiation of the 
Manchester POLE crime graph covering August 2017, originally hosted as a Neo4j dump file at:
https://github.com/neo4j-graph-examples/pole

It encodes **369 persons** connected through five social relationship 
types and linked to crimes, objects, locations, and events.

---

## Setup Instructions

### Prerequisites
- Neo4j Desktop (v5.x or later) — https://neo4j.com/download/
- Neo4j Graph Data Science plugin (v2026.03.0 or compatible)
- Neo4j APOC plugin (optional but recommended)

---
### Step 1 — Import the Database Dump File (pole-50.dump)
### Step 2 — Verify GDS is Installed
### Step 3 — Run the Cypher Queries

Open `gml_queries.cypher` in Neo4j Browser and execute queries 
**one at a time, top to bottom**. The file is divided into three 
clearly labelled sections:

| Section | Task | Description |
|---------|------|-------------|
| GML-1 | Community Detection | Louvain algorithm — finds criminal clusters |
| GML-2 | Node Classification | Repeat offender prediction via FastRP + Random Forest |
| GML-3 | Link Prediction | Hidden connection discovery via Hadamard + Random Forest |

> **Important:** GML-1 must complete before GML-3 because the 
> cross-community analysis in GML-3 depends on the `communityId` 
> property that GML-1 writes to every Person node.

---

## Authors

**Muhammad Wasiq Shaikh**
School of Science & Engineering, Habib University
ms09205@st.habib.edu.pk

**Zainab Huzefa**
School of Science & Engineering, Habib University
zh08968@st.habib.edu.pk

---