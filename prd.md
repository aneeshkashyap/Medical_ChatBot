# Product Requirements Document

## Medical Information Chatbot

---

# 1. Product Overview

The Medical Information Chatbot is a healthcare knowledge retrieval system for educational use.
It accepts natural language medical queries, maps them to structured dataset knowledge, and returns a prescription-style report with causes, symptoms, care guidance, and trusted references.
It is not a clinical diagnosis system.

---

# 2. Implementation Status (As Built)

Based on the current implementation in [chatbot.py](chatbot.py), the product is substantially developed.

## 2.1 Completed

* Natural language query input for definition, cause, symptoms, and free-text symptom descriptions
* Input normalization (lowercase, punctuation removal, stopword filtering)
* Intent detection (definition, cause, symptom, symptom-description style)
* Weighted similarity ranking:
	* keyword score
	* Jaccard similarity
	* fuzzy similarity
	* category boost
	* intent boost
* Symptom-based condition inference for statements such as personal symptom narratives
* Prescription-style structured table output with:
	* patient name
	* probable condition
	* cause
	* symptoms
	* solution/care plan
	* urgent warning signs
	* source links
* Research-depth transparency fields:
	* evidence coverage
	* research method
	* score breakdown
	* top related matches
	* analysis timestamp
* Safety disclaimer included in every report
* Query caching for repeated user inputs

## 2.2 Partially Completed

* Verified sources are category-mapped reference links, not citation-per-answer validation
* Performance target is implied for small datasets but not benchmarked by automated tests

## 2.3 Not Yet Implemented

* Automated evaluation dashboard for accuracy and reliability metrics
* Model-based clinical reasoning or triage classifier
* Multi-turn context memory beyond current single-query focus

---

# 3. Problem Statement

Users often encounter unreliable health information online.
The product must provide clear, structured, and source-backed general medical information with strong safety boundaries.

---

# 4. Objectives

* Provide reliable educational healthcare answers in conversational form
* Present outputs in a readable prescription-style structure
* Show transparent matching logic for research analysis
* Preserve clear safety messaging and limitations

---

# 5. Target Users

* Students learning health topics
* General users seeking basic health information
* Researchers evaluating rule-based healthcare chat systems

---

# 6. Functional Requirements (Current Scope)

## 6.1 Query Handling

The system shall accept:

* definition questions (example: What is malaria)
* cause questions (example: What causes ulcer)
* symptom questions (example: Symptoms of dengue)
* free-text symptom descriptions (example: My stomach burns after eating spicy food)

## 6.2 Name-Aware Prescription Output

The system shall:

* request patient name at startup
* allow patient name updates using name command pattern
* include patient name in every generated prescription table

## 6.3 Ranking and Inference

The system shall compute candidate scores using weighted rule-based ranking with category and intent boosts.
For symptom descriptions, it shall infer likely condition from symptom-answer similarity.

## 6.4 Structured Report Generation

The system shall output a terminal-friendly prescription table including:

* patient name
* patient input
* probable condition
* confidence
* description
* cause
* symptoms
* solution/care plan
* urgent care warning
* source list
* research-depth metadata
* medical safety note

---

# 7. Non-Functional Requirements

## Reliability

Responses shall come from dataset-backed entries and mapped source categories.

## Transparency

Reports shall include score and method details for reproducibility.

## Safety

Every output shall include non-diagnostic disclaimer and urgent-care indicators.

## Performance

Target response latency: under 2 seconds for current dataset size in local execution.

---

# 8. Data Requirements

Dataset columns required:

* Question
* Answer
* Category

Coverage should include, at minimum:

* definitions
* causes
* symptoms
* category labels for source/care mapping

---

# 9. Risks and Limitations

* Limited by dataset depth and quality
* Rule-based scoring may mis-rank edge phrasing
* Not suitable for diagnosis or emergency decision-making

---

# 10. Future Enhancements

* Per-answer evidence citations and confidence calibration
* Automated quality evaluation suite and benchmark reporting
* Larger medical corpus integration
* Contextual multi-turn follow-up handling

---

# 11. Success Metrics

* Top-1 match relevance rate
* User-rated clarity of prescription table
* Coverage completeness for common conditions
* Safety compliance rate (disclaimer and red-flag presence)

---

# 12. Ethical and Safety Requirements

The system must remain educational and non-diagnostic.
It must avoid promising cure guarantees and direct users with severe symptoms to professional medical care.

---
